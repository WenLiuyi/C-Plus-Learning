上一篇中阐述了本实验中物理内存、虚拟内存的管理方式。

- 物理内存：页控制块数组中，每一项代表一页物理内存；使用基于双向链表结构的空闲链表`page_free_list`管理空闲页面，实现页控制块的**申请和释放**，此时，`page_free_list`可视作一个资源池。
- 虚拟内存：当使用`kuseg`地址空间的虚拟地址访问内存时，CPU会**通过TLB将其转换为物理地址**；当TLB中查询不到对应的物理地址时，就会触发TLB Miss异常。这时将跳转到异常处理函数，执行TLB重填。

现在，我们进入创建和调度进程的环节。

## 进程

> 我们编写的代码是一个存储在硬盘的静态文件，通过编译后生成⼆进制可执行文件；当我们运行该可执行文件时，它会被装载到内存中，接着 CPU 会执⾏程序中的每⼀条指令，那么这个运⾏中的程序，就被称为**进程**。 进程的定义：**进程是具有独立功能的程序在⼀个数据集合上运⾏的过程，是系统进行资源分配和调度的⼀个独立单位**。

在本实验中未实现线程，因此进程同时是基本的分配单元和执行单元。

### 进程控制块`Env`

> 统通过**进程控制块PCB**描述进程的基本情况和运行状态，进而控制和管理进程。它是进程存在的**唯一标识**，包含以下信息： 
>
> 1. 进程描述信息：进程标识符，用户标识符；
> 2. 进程控制和管理信息：进程当前状态，进程优先级；
> 3. 进程资源分配清单：有关内存地址空间或虚拟地址空间的信息，所打开⽂件的列表和所使⽤的 I/O 设备信息；
> 4. CPU相关信息： 当进程切换时，CPU寄存器的值都被保存在相应PCB中，以便CPU重新执⾏该进程时能从断点处继续执⾏。
>
> `PCB` 通常是通过**链表**的⽅式进⾏组织，把具有**相同状态的进程链在⼀起，组成各种队列**。

```c
struct Env {
	struct Trapframe env_tf;	 // 进程切换前，保存的当前进程上下文环境
	LIST_ENTRY(Env) env_link;	 // 类似于pp_link，用于构造：空闲进程链表env_free_list
	u_int env_id;			 // 唯一进程标识符
	u_int env_asid;			 // 进程的ASID
	u_int env_parent_id;		 // 父进程ID
	u_int env_status;		 // 进程状态：ENV_FREE；ENV_NOT_RUNNABLE；ENV_RUNNABLE
	Pde *env_pgdir;			 // 进程页目录的内核虚拟地址
	TAILQ_ENTRY(Env) env_sched_link; // 用于构造：调度队列 env_sched_list
	u_int env_pri;			 // 进程的优先级

	......
};
```

> 补充：
>
> 1. `Trapframe`结构体：在发生进程调度或陷入内核时，保存当前进程的上下文环境。
>
> ```c
> struct Trapframe {
> 	/* Saved main processor registers. */
> 	unsigned long regs[32];
> 
> 	/* Saved special registers. */
> 	unsigned long cp0_status;
> 	unsigned long hi;
> 	unsigned long lo;
> 	unsigned long cp0_badvaddr;
> 	unsigned long cp0_cause;
> 	unsigned long cp0_epc;
> };
> ```
>
> 2. `env_status`字段取值：
>    * `ENV_FREE`：当前进程处于空闲状态，位于空闲链表中；
>    * `ENV_NOT_RUNNABLE`：当前进程处于**阻塞状态**，转为就绪状态后，才能被CPU调度；
>    * `ENV_RUNNABLE`：当前进程处于**执行状态/就绪状态**。
> 3. `env_sched_link`使用结构体`TAILQ_ENTRY`，实现双端队列。支持头部、尾部的插入和取出。



### 进程的标识

#### 进程标识符

操作系统通过**进程标识符**来识别进程，对应`Env`结构体中的`env_id`域，在进程创建时被赋予。

> 进程标识符的生成函数如下：
>
> ```c
> #define LOG2NENV 10
> u_int mkenvid(struct Env *e) {
> 	static u_int i = 0;
> 	return ((++i) << (1 + LOG2NENV)) | (e - envs);
> }
> ```
>
> 该函数在`env_alloc`中被调用：`e->env_id=mkenvid(e);`用于初始化进程块时，分配标识符



#### 进程的ASID

`env_asid` 域记录进程的 ASID，即**进程虚拟地址空间的标识**。

> **那么，为什么要额外引入ASID呢？**
>
> 系统中**并发执行多个拥有不同虚拟地址空间的进程，分别具有不同的页表；**而 CPU 的 MMU 使用 TLB 缓存虚拟地址映射关系，**不同页表拥有不同的虚拟地址映射**。
>
> 因此，**不同进程的虚拟地址，可以对应相同的虚拟页号**。
>
> 当 CPU 切换页表，TLB 中仍可能缓存有之前页表的虚拟地址映射关系，为了避免这些无效映射关系导致错误的地址翻译，**早期操作系统实现在 CPU 每次切换页表时，无效化所有 TLB 表项。**
>
>
> 然而，这种实现导致频繁的 TLB Miss，影响处理器性能。现代的 CPU 及操作系统，采用ASID 解决上述问题。ASID 用于标识虚拟地址空间，同时并发执行的多个进程具有不同 ASID，以方便 TLB 标识其虚拟地址空间。

Lab2中提到：在`4Kc`中，TLB由⼀组Key +两组Data 组成，**构建映射：`< VPN, ASID >` TLB---> `< PFN, N, D, V, G >`**。对于每一个进程，都有ASID 标识的虚拟地址空间下的**一套独立 TLB 缓存**。因此，切换页表时，操作系统不必再清空所有TLB表项。

那么，**ASID何时分配和回收呢？**

1. **初始化进程块时创建**：在`env_alloc`函数中有：

   `if(asid_alloc(&e->env_asid)==-E_NO_FREE_ENV) return -E_NO_FREE_ENV;`

   > ASID分配函数：`asid_alloc`（采用**位图法**：**共256个可⽤的ASID,0~7位表示**）
   >
   > ```c
   > static int asid_alloc(u_int *asid) {
   > 	for (u_int i = 0; i < NASID; ++i) {
   > 		int index = i >> 5;
   > 		int inner = i & 31;	//inner为i的低5位
   > 	//定义:static uint32_t asid_bitmap[NASID / 32] = {0};
   > 	//asid_bitmap每个元素32位,对应32个ASID的分配状态
   > 		if ((asid_bitmap[index] & (1 << inner)) == 0) {	//未分配
   > 			asid_bitmap[index] |= 1 << inner;	//标为已分配
   > 			*asid = i;
   > 			return 0;
   > 		}
   > 	}
   > 	return -E_NO_FREE_ENV;
   > }
   > ```

2. 