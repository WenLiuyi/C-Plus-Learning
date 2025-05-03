# Ray集群介绍+verl Ray API

## 现代计算机体系结构

现代计算机体系结构如下：

* **多核**：一台计算机上有多颗CPU，每个 CPU 有多个计算核心。CPU内部有缓存结构，外部有主存。
* **集群**：多台计算机通过高速网络互联，每台计算机上配有至少一块高速网卡。使得不同节点之间互相访问数据就像在单个节点一样。
* **异构计算**：CPU 和主存通常被称为主机（Host），各类专用的加速器被称为设备（Device）。当前基于 GPU 的异构计算是主流，GPU 有区别于 CPU 的芯片微架构和编译软件栈。
  * 软件层面：GPU 提供了 CUDA编程接口；
  * 硬件层面：GPU 有很多个专用计算核心，和 GPU 上的存储。

![../_images/computer-arch.svg](https://scale-py.godaai.org/_images/computer-arch.svg)

## 并行程序设计方法：PCAM

如何设计软件和算法，使得程序可以并行运行在多核或者集群上？PCAM共包括4个步骤：

* **切分**：将整个问题切分为多个子问题或子任务，既包括计算部分也包括数据部分；
* **通信**：不同子任务之间通信方式，需要包括通信的数据结构、通信算法；
* **聚集**：考虑到当前所拥有的硬件性能和编程难度，将上面两步进一步整合，将细粒度的任务整合成更高效的任务；
* **分发**：将整合好的任务分发给多个处理器。

> 举个栗子：有一个超大矩阵，矩阵大小为 M×M，这个矩阵大到无法放在单个计算节点上计算，现在想获取这个矩阵的最大值。设计并行算法时，可以考虑如下思路：
>
> - 将矩阵切分成子矩阵，每个子矩阵 m×m 大小，在**每台计算节点上执行 `max()` 函数**求得子矩阵的最大值；
> - 将**每个子矩阵的最大值汇集到一个计算节点**，在该节点再次执行一下 `max()` 求得整个矩阵的最大值；

![../_images/pcam.svg](https://scale-py.godaai.org/_images/pcam.svg)

### 案例：MapReduce

Google在2004年提出的MapReduce是一种经典的大数据并行计算范式。其中主要涉及四个阶段：

- 切分（Split）：将大数据切分成很多份小数据，**每份小数据可以在单个 Worker 上计算**。
- 映射（Map）：**对每个小数据执行 Map 操作**，Map 是一个函数映射，程序员需要**自定义 Map 函数，Map 函数输出一个键值对（Key-Value）**。在词频统计的例子中，每出现一个词，计 1 次，Key 是词，Value 是 1，表示出现 1 次。
- 交换（Shuffle）：**将相同的 Key 归结到相同的 Worker 上**。这一步涉及数据交换。词频统计的例子中，将相同的词发送到同一个 Worker 上。
- 聚合（Reduce）：**所有相同的 Key 进行聚合操作**，程序员需要**自定义 Reduce 函数**。词频统计的例子中，之前 Shuffle 阶段将已经将相同的 Key 归结到了一起，现在只需要将所有词频求和。

![../_images/map-reduce.svg](https://scale-py.godaai.org/_images/map-reduce.svg)

### 性能指标

#### FLOPs

FLOPS 指**每秒钟能够完成多少次浮点计算**。如果进行一个 n 维向量加法：a+b，所需的浮点计算次数为 n。将浮点计算次数除以时间，就是 FLOPS。

#### 加速比

衡量并行相对于串行执行时间的缩短程度：加速比=$\frac{t_s}{t_p}$，其中 $t_s$ 为串行程序执行时间，$t_p$ 为并行程序执行时间。

* **效率**：效率=$\frac{加速比}{N}$。其中 N 为并行程序所使用的计算核心的数目。

当加速比为 N 时，串行程序可以被线性拓展到多个计算核心上，可以说并行程序获得了**线性加速比**，即理想情况。现实中，并行程序需要有调度器将不同的任务分发到多个 Worker 上，多个 Worker 之间需要通信，以及数据需要在多个 Worker 之间需要同步，这些步骤都会浪费时间。



## Ray

### Ray结构

Ray最初为强化学习设计。

当前 Ray 主要由底层的 Ray Core 和上层的各类 Ray AI (Artificial Intelligence) 生态组成：

* Ray Core 是一系列底层 API, 可以将 Python 函数或者 Python 类等计算任务**横向扩展到多个计算节点上**；
* 在 Ray Core 之上，Ray 封装了一些面向数据科学和人工智能的库（Ray AI Libraries），可以进行数据的处理（Ray Data）、模型训练（Ray Train）、模型的超参数调优（Ray Tune），模型推理服务（Ray Serve），强化学习（RLib）等。



##### Ray Core API

Ray Core的核心API如下：

* **Task**：面向**函数**的接口，该函数可在集群中分布式执行；
* **Actor**：面向**类**的接口，该类可在集群中分布式执行；
* **Object**：分布式对象（不可变），用于在**Task和Actor之间传递数据**。

![../_images/ray-apis.svg](https://scale-py.godaai.org/_images/ray-apis.svg)

#### 分布式函数（Remote Function）：`@ray.remote` 装饰器

通过Ray API定义的Task即远程函数，可以运行在远程的Ray集群上。远程函数是**无状态**的：只依赖于函数的输入和输出，不依赖函数作用域之外的中间变量。那么如何将 Python 函数横向扩展到 Ray 集群上？

* **启动Ray集群**：可使用[ray.init()函数](https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html)，启动一个**单节点的Ray集群**，运行在执行这个 Python 任务的计算机上。例如：

  ```python
  if ray.is_initialized:
      ray.shutdown()
  ray.init(logging_level=logging.ERROR)
  ```

通过几个栗子演示。假设使用原生的Python定义一个fibonacci函数，想让这个 Python 函数被 Ray 分布式执行，只需要**在函数上增加一个 `@ray.remote` 装饰器**。

```python
# fibonacci函数
def generate_fibonacci(sequence_size):
    fibonacci = []
    for i in range(0, sequence_size):
        if i < 2:
            fibonacci.append(i)
            continue
        fibonacci.append(fibonacci[i-1] + fibonacci[i-2])
    return len(fibonacci)
# 在函数上增加一个 @ray.remote 装饰器
@ray.remote
def generate_fibonacci_distributed(sequence_size):
    return generate_fibonacci(sequence_size)
```

**作为 Ray 的使用者，无需关心 Task 在 Ray 集群中是如何被分布式执行的，也不需要了解这个 Task 被调度到哪些计算节点**。所有这些分布式执行的细节都被 Ray 所隐藏，或者说 Ray 帮我们做了底层的分布式与调度这些工作。

使用 Ray 进行分布式扩展，函数可并行地在多个 CPU 核心上执行：

```python
# 使用 Ray 进行分布式扩展
def run_remote(sequence_size):
    results = ray.get([generate_fibonacci_distributed.remote(sequence_size) for _ in range(os.cpu_count())])
    return results
```



> 原生Python函数和Ray的区别：
>
> * **调用方式**：
>   * 原生Python函数：使用 `func_name()` 调用；
>   * 使用 Ray 时：函数定义增加 `@ray.remote` 装饰器，调用时使用 [`func_name.remote()`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) 的形式。
> * **返回值**：
>   * 使用 Ray 时：`func_name.remote()` 返回值是 `ray.ObjectRef` **类型的对象**，`ray.ObjectRef` 并不是一个具体的值，而是一个 Future（尚未完成但未来会完成的计算），需要**使用 [`ray.get()`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html) 函数获取该调用的实际返回值**。
> * **执行方式**：
>   * 原生Python函数：调用形成阻塞，等待结果返回才进行后续计算（**同步执行**）；
>     * 使用 Ray 时：**异步执行**（`func_name.remote`非阻塞；`ray.get(ObjectRef)`阻塞）
>     * 立即返回一个 `ray.ObjectRef`，调用者不需要等待这个函数的计算真正执行完，函数的计算是在后台某个计算节点上执行的；
>     * `ray.get(ObjectRef)` 会等待后台计算结果执行完，将结果返回给调用者。



#### 分布式对象（Remote Object）存储：`ray.put()` 与 `ray.get()`

Ray 分布式计算中涉及共享数据可被放在分布式对象存储中，这些数据被称为**远程对象**。我们可以使用 [`ray.get()`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.get.html) 和 [`ray.put()`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.put.html) 读写这些远程对象。与内存中的 Python 对象实例不同，Remote Object 是不可原地直接更改的。

操作 Remote Object 主要有 `ray.put()` 和 `ray.get()` 两个 API：

- `ray.put()` ：把**某个计算节点中的对象数据进行序列化**，并将其**写入到 Ray 集群的分布式对象存储中**，返回一个 `RefObjectID`（`RefObjectID` 是**指向这个 Remote Object 的指针**）。我们可以通过引用这个 `RefObjectID`，在 Remote Function 或 Remote Class 中分布式地使用这个数据对象。
- `ray.get()` ：使用 `RefObjectID` 把数据从分布式对象存储中拉取回来，并进行**反序列化**。
- ![../_images/put-get-object-store.svg](https://scale-py.godaai.org/_images/put-get-object-store.svg)

举个栗子：

```python
def create_rand_tensor(size: Tuple[int, int, int]) -> torch.tensor:
    return torch.randn(size=(size), dtype=torch.float)

torch.manual_seed(42)

# 写入数据：put(创建 16 个张量，每个张量大小为 (X, 8, 8))
tensor_obj_ref_list = [ray.put(create_rand_tensor((i, 8, 8))) for i in range(1, 16)]
# 读取数据：get
val = ray.get(tensor_obj_ref_list[0])
```

##### 修改数据

Remote Ojbect 中的数据是不可修改的（Immutable），即无法对变量原地更改。在单机上，我们可以对变量进行赋值；但**在 Ray 中，我们无法原地更改 Remote Object 的值**。

如果想使用新数据，应该使用 Remote Function 或者 Remote Class 对 Remote Object 进行转换操作，**生成新的 Remote Object**。

```python
@ray.remote
def transform_tensor(tensor: torch.tensor) -> torch.tensor:
    return torch.transpose(tensor, 0, 1)
# 使用Remote Function更改数据
transformed_object_list = [transform_tensor.remote(t_obj_ref) for t_obj_ref in tensor_obj_ref_list]
```

##### 传递参数：通过`RefObjectID`

1. **直接传递**：在 Task 或者 Actor 的函数调用时，将 `RefObjectID` 作为参数传递进去。

   ```python
   @ray.remote
   def echo(x):
       print(f"current value of argument x: {x}")
       return x
   
   x = list(range(5))
   # `x_obj_ref` 是一个 `RefObjectID`
   x_obj_ref = ray.put(x)
   # 直接将RefObjectID作为参数传递，echo()这个 Remote Function 将自动从 `x_obj_ref` 获取 `x` 的值，该过程称为：自动反引用
   ray.get(echo.remote(x_obj_ref))
   **复杂数据结构**：如果 `RefObjectID` 被包裹在一个复杂的数据结构中，Ray 并不会自动获取 `RefObjectID` 对应的值，即反引用并不是自动的。
   ```

​	输出：`(echo pid=22623) current value of argument x: [0, 1, 2, 3, 4]`

2. **复杂数据结构**：如果 `RefObjectID` 被包裹在一个复杂的数据结构中，Ray 并不会自动获取 `RefObjectID` 对应的值，即反引用并不是自动的。

   ```python
   ray.get(echo.remote({"obj": x_obj_ref}))	# 包裹在一个 dict 中
   ray.get(echo.remote([x_obj_ref]))					# 包裹在一个 list 中
   ```

   输出：

   ```python
   (echo pid=70963) current value of argument x: {'obj': ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000010000000)}
   (echo pid=70963) current value of argument x: [ObjectRef(00ffffffffffffffffffffffffffffffffffffff0100000010000000)]
   ```

##### 底层实现

1. Ray 集群的**每个计算节点，都有一个基于共享内存的对象存储**。

2. 当某个 Remote Object 的数据量较小时（<= 100 KB），它会被存储在**计算节点进程内存**中；当数据量较大时，它会被存储在**分布式的共享内存**中；当集群的共享内存的空间不够时，数据会被**外溢（Spill）到持久化的存储上**，比如硬盘或者S3。



#### 分布式类（Actor）

举个栗子：

1. Ray 的 Remote Class 也使用 [`ray.remote()`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) 来装饰；

   ```python
   @ray.remote
   class Counter:
       def __init__(self):
           self.value = 0
   
       def increment(self):
           self.value += 1
           return self.value
   
       def get_counter(self):
           return self.value
   ```

2. 初始化一个实例：在类名 `Counter` 后面加上 `remote()`，即创建一个分布式的 Actor;

   ```python
   counter = Counter.remote()
   ```

3. 调用实例的函数：加上 `remote()`，即`对象实例.函数名.remote()`；

   ```python
   obj_ref = counter.increment.remote()
   print(ray.get(obj_ref))
   ```

可以用同一个类创建不同的 Actor 实例：**不同 Actor 实例的成员函数调用可以并行化执行；同一个 Actor 的成员函数调用顺序执行。**

```python
# 创建 10 个 Actor 实例
counters = [Counter.remote() for _ in range(10)]

# 对每个 Actor 进行 increment 操作
# 这些操作可以分布式执行
results = ray.get([c.increment.remote() for c in counters])
print(results)
```

> Actor编程模型：分布式编程的范式，基本要素是 **Actor 实例**，即每个 Actor 对象都是唯一的。可以把单个 Actor 实例理解成单个带地址信息的进程。
>
> * Actor 存储的状态数据只能由 Actor 自己来管理，不能被其他 Actor 修改；
> * **消息驱动**：给某个 Actor 发送消息，它就会对该消息进行响应，修改自身的状态或者继续给其他 Actor 发送消息。
> * 对同一个 Actor 多次发送同样请求，多次请求是顺序执行的。

##### 栗子：Actor Pool

实践上，经常创建一个 Actor 资源池（Actor Pool），[`ActorPool`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.html) 有点像 `multiprocessing.Pool`，包含多个 Actor，每个 Actor 功能一样，而且可以分布式地在多个计算节点上运行。

```python
from ray.util import ActorPool
# 定义一个Actor
@ray.remote
class PoolActor:
    def add(self, operands):
        (a, b) = operands
        return a + b

    def double(self, operand):
        return operand * 2

# 创建3个Actor实例
a1, a2, a3 = PoolActor.remote(), PoolActor.remote(), PoolActor.remote()
# 将创建的 Actor 添加至 ActorPool 中
pool = ActorPool([a1, a2, a3])
```

如果我们想调用 `ActorPool` 中的 Actor，可以使用 [`map(fn, values)`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.map.html) 和 [`submit(fn, value)`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.submit.html) 方法。

* `map()` ：`values` 是一个列表，让函数**并行地分发给多个 Actor 处理**；
* `submit()`： `value` 是单个值，**每次从 `ActorPool` 中选择一个 Actor 执行**。
  * `submit()` 的 `value` 参数只能是单个对象，不能是参数列表，如果想传入多个参数，可以把参数包裹成元组。

`fn` 是一个 Lambda 表达式，或者说是一个匿名函数。这个 Lambda 表达式有两个参数：`actor` 和 `value`，`actor` 是`ActorPool` 中的 Actor，第二个参数是函数的参数。

```python
pool.map(lambda a, v: a.double.remote(v), [3, 4, 5, 4])

pool.submit(lambda a, v: a.double.remote(v), 3)
pool.submit(lambda a, v: a.double.remote(v), 4)
```

`map()` 和 `submit()` 将计算任务提交到了 `ActorPool` 中，`ActorPool` 并不是直接返回结果，而是异步地分发给后台不同的 Actor 去执行。需要使用 [`get_next()`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.util.ActorPool.get_next.html) 阻塞地返回结果。

```py
try:
    print(pool.get_next())
    print(pool.get_next())
    print(pool.get_next())
except Exception as e:
    print(e)
```

结果为：

```bash
6
8
10
```



### Ray集群

Ray 集群由一系列计算节点组成，包括两类关键的节点：**头节点**（Head）和**工作节点**（Worker）。这些节点可以部署在虚拟机、容器或者是裸金属服务器上。

头节点额外包括：GCS，即Ray 集群的全局元数据管理服务；负责存储和管理诸如哪个 Actor 被分配到哪个计算节点等元数据信息。这些**元数据被所有 Worker 共享**。

每个节点包括一个**Driver：执行程序的入口点，指的是Python 的 `__main__` 函数**。通常，`__main__` 在运行时不应该执行大规模计算，而是负责将 Task 和 Actor 调度到具备足够资源的 Worker 上。

![../_images/ray-cluster.svg](https://scale-py.godaai.org/_images/ray-cluster.svg)

在 Ray 分布式计算环境中，所有节点上都运行着一些关键进程。

* **Raylet**：**每个计算节点上运行着一个 Raylet**， Raylet 被多个 Worker 进程所共享。Raylet 主要包含两个组件：一个是**调度器**，它负责资源管理和任务分配；另一个是**基于共享内存的对象存储**，它负责本地数据存储，各个计算节点上的对象存储共同构成了 Ray 集群的分布式对象存储。

* **Worker**：**每个计算节点上运行着一个或多个 Worker 进程**，这些进程负责执行计算任务。Worker 进程可以是无状态的，意味着它们可以反复执行 Task 对应的任务；它们也可以是有状态的 Actor，即执行远程类的方法。**默认情况下，Worker 的数量等于其所在计算节点的 CPU 核心数**。



启动Ray集群：如果Python 代码中使用 `ray.init()` 方式，仅在本地启动了一个单机的 Ray 集群。实际上，Ray 集群包括头节点和工作节点，应该分别启动。先在头节点启动：

```bash
ray start --head --port=6379
```

启动工作节点：

```bash
ray start --address=<head-node-address>:<port>
```

通过`ray up example.yaml`启动：接收 yaml 文件作为参数，在 yaml 文件里定义好头节点地址、工作节点地址。

```yaml
cluster_name: default

provider:
    type: local
    head_ip: YOUR_HEAD_NODE_HOSTNAME
    worker_ips: [WORKER_NODE_1_HOSTNAME, WORKER_NODE_2_HOSTNAME, ... ]
```

> Ray 的头节点暴露三个端口号，默认分别是 6379, 8265, 10001。
>
> 1. 启动 Ray 时，设置了 Ray 头节点的端口号，默认为 **6379**，是**头节点和工作节点之间通信的端口**；
> 2. Ray 头节点启动后，提供了一个 Ray 仪表盘端口号，默认为 8265，可用来接收 Ray 命令行提交的作业；
> 3. 此外，还有一个端口 10001，默认为 `ray.init()` 连接时使用。

#### 计算资源与资源组

Ray 可以管理计算资源，包括 CPU、内存和 GPU 等各类加速器。这里的计算资源是逻辑上的，逻辑资源与物理上的计算资源相对应。**Ray 集群的各个节点启动时会探测物理计算资源，并根据一定规则映射为逻辑上的计算资源。**默认规则如下：

- CPU：每个节点中的物理 CPU 个数（`num_cpus`）

- GPU：每个节点中的物理 GPU 个数（`num_gpus`）

- 内存：每个节点可用内存的 70%（`memory`）

  可自行指定：

  ```bash
  ray start --num-cpus=32 --num-gpus=4

Ray集群支持**自动缩放**，指的是满足 Task 或 Actor 代码中定义的计算资源请求（比如，`task.options()` 请求的计算资源），而不是根据计算节点的资源实际利用情况自动缩放。主要面向以下场景：

- 当 Ray 集群的资源不够时，创建新的工作节点。
- 当某个工作节点闲置或者无法启动，将该工作节点关闭。

##### 资源需求

默认情况下：

* Ray Task使用1个逻辑CPU，既用于任务调度，也用于执行计算任务；

* Ray Actor使用1个逻辑CPU进行任务调度，0 个 CPU 运行计算任务。

  * 如果不做设置，可能造成 Ray Actor 不需要计算资源的假象，导致大量 Actor 被调度到同一个计算节点上。可进行指定：

    ```python
    @ray.remote(num_cpus=4)
    def func():
        ...
    
    @ray.remote(num_cpus=16, num_gpus=1)
    class Actor:
        pass
      
    # 或者：
    func.options(num_cpus=4).remote()
    ```



##### 资源组（Placement Group）

允许用户**原子地**使用集群上多个节点的计算资源：资源要么全部分配给用户，要么完全不分配，不会出现只分配部分资源的情况。主要适用以下场景：

- **组调度**：一个作业需要一组资源，这些资源需要协同工作以完成任务。要么分配，要么不分配。如果只分配给这个作业部分资源，将无法完成整个任务。
  - 例如在大规模分布式训练中：可能需要多台计算节点和多块GPU，这时可以在Ray集群中申请并分配这些资源。
- **负载均衡**：作业需要在多个节点上进行负载均衡，每个节点承担一小部分任务。Placement Group可以确保作业尽量分散到多个计算节点上。
  - 例如在分布式推理场景中：如果一个作业需要8块GPU，每个GPU负责加载模型并独立进行推理，为了实现负载均衡，应该将作业调度到8个计算节点上，每个节点使用1块GPU。这样做的好处是，如果一个节点发生故障，不会导致整个推理服务不可用，因为其他节点仍然可以继续工作。

关键概念：

* **资源包（Bundle）**：**一个键值对，定义所需的计算资源**，比如 `{"CPU": 2}`，或 `{"CPU": 8, "GPU": 4}`。**一个 Bundle 必须可以调度到单个计算节点**；比如，一个计算节点只有 8 块 GPU，`{"GPU": 10}` 是不合理的。
  * 多个 Ray Task 或 Actor 可以运行在同一个 Bundle 上；任何使用同一个 Bundle 的 Task 或 Actor 将一直运行在该计算节点上。
* **资源组（Placement Group）**：Placement Group 是**一组 Bundle**。比如，`{"CPU": 8} * 4` 向 Ray 集群申请 4 个 Bundle，每个 Bundle 预留 8 个 CPU。

举个完整栗子：

```python
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray
# 启动ray集群
ray.init()
print('''Available Resources: {}'''.format(ray.available_resources()))

@ray.remote(num_gpus=2)
def gpu_task():
    print("GPU ids: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))

# 创建 Placement Group：包括一个Bundle
pg = placement_group([{"CPU": 16, "GPU": 2}])
# 等待 Placement Group 创建成功
ray.get(pg.ready(), timeout=10)
# 也可以使用 ray.wait
ready, unready = ray.wait([pg.ready()], timeout=10)
print('''Placement Group: {}'''.format(placement_group_table(pg)))

# 将 Ray Task 调度到这个 Placement Group
ray.get(gpu_task.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
).remote())

# 删除这个 Placement Group
remove_placement_group(pg)
```

> `placement_group()` 接收 `strategy` 参数：
>
> * `STRICT_PACK`：所有 Bundle 都必须调度到单个计算节点。
>
> * `PACK`（默认策略）：**所有 Bundle 优先调度到单个计算节点**，如果无法满足条件，再调度到其他计算节点，
>
>   ![../_images/pg-pack.svg](https://scale-py.godaai.org/_images/pg-pack.svg)
>
> * `STRICT_SPREAD`：每个 Bundle 必须调度到不同的计算节点。
>
> * `SPREAD`：每个 Bundle 优先调度到不同的计算节点，如果无法满足条件，有些 Bundle 可以共用一个计算节点。
>
>   ![../_images/pg-spread.svg](https://scale-py.godaai.org/_images/pg-spread.svg)
>
>   对比：
>
>   * `STRICT_PACK` 和 `PACK` 保证了数据的**局部性**，计算任务可以快速访问本地的数据；
>   * `STRICT_SPREAD` 和 `SPREAD` 使得计算更好地负载均衡。



#### Ray作业

Ray 作业指的是用户编写的，基于 Task、Actor 或者 Ray 各类生态（Ray Train、Ray Tune、Ray Serve、RLlib 等）的**具体的计算任务**。主要包括三种作业提交方式：

1. **Ray Jobs 命令行**：`RAY_ADDRESS` 根据头节点的地址设定； `--working-dir` 为工作目录，Ray 会将该目录下的内容打包，分发到 Ray 集群各个节点；ENTRYPOINT指的是需要执行的 Python 脚本，本例中，是 `python script.py`.

   ```bash
   RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir ./ -- python script.py
   ```

   依赖管理：启动作业时，设置 `--runtime-env-json`，原理是为每个作业创建一个独立的虚拟环境。

2. **Python SDK**：将提交作业的各类参数写在 Python 代码中，执行 Python 代码来提交作业。

   ```py
   import time
   from ray.job_submission import JobSubmissionClient, JobStatus
   
   client = JobSubmissionClient("http://127.0.0.1:8265")
   # submit_job()方法的作业提交是异步的：调用此方法后，Ray 会马上返回作业的 ID
   job_id = client.submit_job(
       entrypoint="python script.py",
       runtime_env={"working_dir": "./"}
   )
   print(job_id)
   
   def wait_until_status(job_id, status_to_wait_for, timeout_seconds=5):
       start = time.time()
       while time.time() - start <= timeout_seconds:
           status = client.get_job_status(job_id)
           print(f"status: {status}")
           if status in status_to_wait_for:
               break
           time.sleep(1)
   
   # wait_until_status() 函数不断向 Ray 集群请求，检查作业的当前状态
   wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
   logs = client.get_job_logs(job_id)
   print(logs)
   ```

3. **Ray客户端**：在 Python 中使用 `ray.init()` 函数，直接指定Ray集群的地址：`ray.init("ray://<head-node-host>:<port>")`。

   在客户端与Ray集群意外断开连接的情况下，Ray会尝试在30秒后重新建立连接。如果重新连接失败，Ray将销毁所有相关的引用。可以通过设置环境变量 `RAY_CLIENT_RECONNECT_GRACE_PERIOD` 来自定义这个重连尝试的时间间隔。



### Ray Data

Ray Data 是一个构建在 Ray Core 之上的数据处理框架，对数据提供了一个抽象类：[`ray.data.Dataset`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.html)，它封装了数据并在上面实现了常见的大数据处理原语。包括：

- 数据的读取：比如读取 Parquet 文件等。
- 对数据的转换（Transformation）操作：比如 [`map_batches()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)。
- 分组聚合操作：比如 [`groupby()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.groupby.html)
- 数据在计算节点间的交换：比如 [`random_shuffle()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.random_shuffle.html) 和 [`repartition()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.repartition.html) 等。

#### `ray.data.Dataset`

`Dataset` 底层的基本单元是 `Block`；`Dataset` 实际上是一个分布式的 `ObjectRef[Block]`。

`Block` 是一个数据结构，它基于Apache Arrow格式构建，这是一种高效率的**列式存储**格式，适用于在内存中处理和操作大量数据。

以下展示了一个由 3 个 `Block` 组成的 `Dataset`：可以使用 `from_*()` API 从其他系统或格式导入成 `Dataset`，比如 [`from_pandas()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.from_pandas.html) 、[`from_spark()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.from_spark.html)。或者使用 `read_*()` API 从持久化的文件系统重读取，比如 [`read_parquet()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.read_parquet.html)、[`read_json()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.read_json.html) 等。

![../_images/dataset-arch.svg](https://scale-py.godaai.org/_images/dataset-arch.svg)

#### 数据读写

Ray Data 使用 **Ray Task 并行地读写数据**：![../_images/dataset-read.svg](https://scale-py.godaai.org/_images/dataset-read.svg)

* 数据加载：

  |      | Parquet          | Text          | CSV          | TFRecord           | 二进制                |
  | ---- | ---------------- | ------------- | ------------ | ------------------ | --------------------- |
  | 方法 | `read_parquet()` | `read_text()` | `read_csv()` | `read_tfrecords()` | `read_binary_files()` |

* 行列裁剪：

  ```python
  import pyarrow as pa
  
  dataset = ray.data.read_parquet(
      dataset_path,
      columns=["passenger_count", "tip_amount", "payment_type"],
      filter=pa.dataset.field("tip_amount") > 6.0
  )
  dataset.show(limit=2)
  ```

* 并行度：各类数据读取方法都可以设置 `parallelism` 参数，来控制底层的并行执行的过程。如果不设置 `parallelism`，Ray Data 通过以下方式试探 `parallelism`：

  1. Ray 获取集群中可用的 CPU 核数；
  2. `parallelism` 被设置为 CPU 核数的 2 倍。如果 `parallelism` 小于 8，则设置为 8；
  3. 估计每个 `Block` 的大小，如果每个 `Block` 平均大于 512 MiB，Ray 增大 `parallelism`，**直到每个 `Block` 小于 512 MiB**。

* 查看数据：...

#### 数据转换

略



## verl Ray API

### 基础执行单元：Worker

![class_diagram_simplified](/Users/lisa/Documents/GitHub/verl/verl/class_diagram_simplified.png)

```python
# 简化后代码：原始代码位于verl/single_controller/base/worker.py
class Worker(WorkerHelper):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        rank = os.environ.get("RANK", None)
        worker_group_prefix = os.environ.get("WG_PREFIX", None)
        if None not in [rank, worker_group_prefix] and 'ActorClass(' not in cls.__name__:
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))
        return instance
    
    def _configure_before_init(self, register_center_name: str, rank: int):
        # rank=0时，配置MASTER_ADDR和MASTER_PORT环境变量，并将该信息存储在self.register_center中
        if rank == 0:
            master_addr, master_port = self.get_availale_master_addr_port()
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }
            if os.getenv("WG_BACKEND", None) == "ray":
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center
                self.register_center = create_worker_group_register_center(name=register_center_name,info=rank_zero_info)
            os.environ.update(rank_zero_info)

    def __init__(self, cuda_visible_devices=None) -> None:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            '_world_size': world_size,
            '_rank': rank,
            '_local_world_size': local_world_size,
            '_local_rank': local_rank,
            '_master_addr': master_addr,
            '_master_port': master_port
        }
        # WorkerMeta仅是包store信息存储在实例对象中
        meta = WorkerMeta(store=store)
        # 将meta(store)信息更新到当前实例的__dict__中并配置环境变量
        self._configure_with_meta(meta=meta)
```



一个栗子：自定义`GPUAccumulator`：继承`Worker`类，假设有4个GPU，则每个GPU实例化一个GPUAccumlator，其成员变量value初始化为GPU rank，然后对所有value执行加1操作。

```python
@ray.remote
class GPUAccumulator(Worker):
    def __init__(self) -> None:
        super().__init__()
        # The initial value of each rank is the same as the rank
        self.value = torch.zeros(size=(1,), device="cuda") + self.rank

    def add(self, x):
        self.value += x
        print(f"rank {self.rank}, value: {self.value}")
        return self.value.cpu()
      
class_with_args = RayClassWithInitArgs(GPUAccumulator)
resource_pool = RayResourcePool([4], use_gpu=True)
workergroup = RayWorkerGroup(resource_pool, class_with_args)
print(workergroup.add(x=1)) # 输出：[tensor([1.]), tensor([2.]), tensor([3.]), tensor([4.])]
```



#### 初始化参数：RayClassWithInitArgs

**`RayClassWithInitArgs`保存通过`@ray.remote`定义的Actor类，以及一些用于异步调用该Actor时所需要的参数**。

```python
class_with_args = RayClassWithInitArgs(GPUAccumulator)
```



#### 资源池：**RayResourcePool**

**RayResourcePool继承自ResourcePool**。

1. **ResourcePool负责存储资源相关的信息**：

* 初始化参数：

  * **process_on_nodes**: 节点进程数列表，表示每个节点上要运行的进程数量

  * **max_colocate_count**: 单个节点上最大并行进程数，默认10

  * **n_gpus_per_node**: 每个节点的GPU数量，默认8

* 关键属性：

  * `_store`: 存储各节点的进程数配置

  * `world_size`: 属性，计算所有节点的总进程数

```python
class ResourcePool:
    def __init__(self, process_on_nodes=None, max_colocate_count: int = 10, n_gpus_per_node=8) -> None:
        if process_on_nodes is None:
            process_on_nodes = []
        self._store = process_on_nodes
        self.max_colocate_count = max_colocate_count
        self.n_gpus_per_node = n_gpus_per_node  # this is left for future huawei GPU that contains 16 GPUs per node

    def add_node(self, process_count):	# 添加新节点到资源池（动态扩展）
        self._store.append(process_count)

    @property
    def world_size(self):
        return sum(self._store)

    def __call__(self) -> Any:
        return self._store

    @property
    def store(self):
        return self._store
		# 获取本地信息
    def local_world_size_list(self) -> List[int]:	# 生成每个进程对应的本地世界大小列表
        nested_local_world_size_list = [[local_world_size for _ in range(local_world_size)] for local_world_size in self._store]
        return [item for row in nested_local_world_size_list for item in row]

    def local_rank_list(self) -> List[int]:		# 生成每个进程的本地rank列表
        nested_local_rank_list = [[i for i in range(local_world_size)] for local_world_size in self._store]
        return [item for row in nested_local_rank_list for item in row]
```

2. **RayResourcePool通过Ray的Placement Group实现资源池的分配**。

```python
class RayResourcePool(ResourcePool):
    def __init__(
        ......
    ) -> None:
        super().__init__(process_on_nodes, max_colocate_count)
        ......

    def get_placement_groups(self, strategy="STRICT_PACK", name=None):
      # 默认使用STRICT_PACK策略：所有 Bundle 都必须调度到单个计算节点
      # (每个bundle包含max_colocate_count个CPU核心)
        if self.pgs is not None:
            return self.pgs		# 缓存已创建的placement groups

        # 生成唯一资源组名称
        pg_name_prefix = name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        # 构建资源bundle配置
        pg_scheme = [[{"CPU": self.max_colocate_count, "GPU": 1} if self.use_gpu else {"CPU": self.max_colocate_count} for _ in range(process_count)] for process_count in self._store]

        lifetime = "detached" if self.detached else None
				# 创建placement groups
        pgs = [placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime) for idx, bundles in enumerate(pg_scheme)]

        ray.get([pg.ready() for pg in pgs])	 # 等待所有资源组就绪

        self.pgs = pgs	# 缓存结果
        return pgs
```



一个栗子：

创建集群：

```python
import ray
from ray.util.placement_group
# 创建包含8GPU、16CPU的Ray集群
ray.init(num_cpus=16, num_gpus=8)
```

创建两个Placement Group，每个Placement Group包含4个GPU和8个CPU。

```python
resource_pool = RayResourcePool(process_on_nodes=[4,4], max_colocate_count=2, use_gpu=True) # 创建资源池
pgs = resource_pool.get_placement_groups() # 创建placement group的列表
```

单个Placement Group创建为：

```python
pg = placement_group(bundles=[{"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1},
                             {"CPU": 2, "GPU": 1}])
```

即`process_on_nodes`指定要创建几个Placement Group，以及每个包含多少GPU；`max_colocate_count`是则bundle中单个GPU最多对应多少个CPU，因为colocate的actor至少要有1个CPU。



#### 资源调度器：RayWorkerGroup

##### 初始化函数：`__init__`

```python
def __init__(
        self,
        resource_pool: RayResourcePool = None,
        ray_cls_with_init: RayClassWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        detached=False,
        worker_names=None,
        ray_wait_register_center_timeout: int = 300,
        **kwargs,
    ) -> None:
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        ......
				# 分离模式：连接已存在的持久化工作者
        if self._is_init_with_detached_workers:
            self._init_with_detached_workers(worker_names=worker_names)
        else:
        # 附着模式：基于资源池创建新工作者（基于resource_pool的信息，启动worker）
            self._init_with_resource_pool(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, bin_pack=bin_pack, detached=detached)
				# ray_cls_with_init.clsz中的某些方法绑定到RayWorkerGroup上
        if ray_cls_with_init is not None:
            self.(self.ray_cls_with_init.cls, func_generator)
```

##### 启动Workers：`_init_with_resource_pool`

```python
def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
				......
    	  # max_collocate_count意味着单个GPU上至多有对应几个CPU
        num_gpus = 1 / resource_pool.max_colocate_count

        rank = -1
        local_world_size = resource_pool.store[0]
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "
            for local_rank in range(local_world_size):
                rank += 1
                # 1. 构造worker需要的配置信息，包括环境变量等；
                # 2. 通过ray_cls_with_init.update_options更新这些配置信息
                # 3. 创建一个worker：
                worker = ray_cls_with_init(placement_group=pg, placement_group_bundle_idx=local_rank, use_gpu=use_gpu, num_gpus=num_gpus)
                self._workers.append(worker)
                self._worker_names.append(name)
				......
```

##### 异步执行：`execute_all_async`

在`_init_with_resource_pool`后，`self._workers`中保存着所有的`worker`。

同步执行：

```python
def execute_all_sync(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))
```

调用异步执行：

```python
def execute_all_async(self, method_name: str, *args, **kwargs):
        length = len(self._workers)
    		# 检查参数是否为列表且长度匹配:
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # 1. 参数分片并执行:遍历每个 worker 的索引i
                result = []
                for i in range(length):
                  '''
                  sliced_args: 从每个位置参数 args 中取出第 i 个元素，组成新的位置参数。
									sliced_kwargs: 从每个关键字参数 kwargs 的值中取出第 i 个元素，组成新的关键字参数。
                  '''
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs))
                return result
				# 2. 如果参数不是分片的: 对每个 worker 使用相同的 args 和 kwargs 调用方法
        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]
```

但是，利用`execute_all_async`来调用worker的不太方便。所以，利用装饰器`register`和`_bind_worker_method`来令调用更加自然。

##### worker方法绑定至workergroup：`_bind_worker_method`

`_bind_worker_method`来自基类`WorkerGroup`，参数包含`user_defined_cls`和`func_generator`。其中`user_defined_cls`就是用户自定义的worker类。



**函数生成器** `func_generator`：**动态生成一个可执行函数**，用于在分布式 Worker 组（`WorkerGroup`）上执行任务。这个生成的函数会按照指定的 **分发（dispatch）、执行（execute）、收集（collect）** 逻辑运行，并支持阻塞和非阻塞模式。

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    '''method_name: 要在 Worker 上调用的方法名（如 "foo"）。
		dispatch_fn: 分发函数，负责将输入参数分发给各个 Worker。
		collect_fn: 收集函数，负责聚合 Worker 返回的结果。
		execute_fn: 执行函数，负责在 Worker 上真正运行方法。
		blocking: 是否阻塞等待结果（True 表示同步，False 表示异步）。'''
    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(self, *args, **kwargs)	# 1. 分发参数
        padding_count = kwargs.pop(_padding_size_key, 0)	# 2. 处理可能的填充
        output = execute_fn(method_name, *args, **kwargs)	# 3. 执行任务
        '''判断是否阻塞等待结果：
        1. 如果 blocking=True，调用 ray.get(output) 等待所有 Worker 完成计算；
        2. 如果 blocking=False，直接返回异步引用。
        '''
        if blocking:
            output = ray.get(output)
        # 4. 收集结果
        output = collect_fn(self, output)
        # 5. 移除填充（如果有）
        if padding_count > 0:
            if isinstance(output, DataProto):
                indices = [i for i in range(len(output))][:-padding_count]
                output = output.select_idxs(indices)
            elif isinstance(output, list):
                output = output[:-padding_count]
        return output

    return func
```



```python
def _bind_worker_method(self, user_defined_cls, func_generator):
        method_names = []
        for method_name in dir(user_defined_cls):	# 遍历类的所有方法
        		# 尝试获取方法并检查是否可调用（callable），跳过不可调用的属性（如 property）
            try:
                method = getattr(user_defined_cls, method_name)
                assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
            except Exception:
                continue
						# 检查方法是否带有特定装饰器标记MAGIC_ATTR
            if hasattr(method, MAGIC_ATTR):
              '''
              获取装饰器设置的属性（attribute），并检查它是否是字典。
							确保属性中包含 dispatch_mode（分发模式）、execute_mode（执行模式）和 blocking（是否阻塞）字段。
              '''
                attribute = getattr(method, MAGIC_ATTR)
                assert isinstance(attribute, Dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, "attribute must contain dispatch_mode in its key"
                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]
                blocking = attribute["blocking"]

                # 获取分发函数（dispatch_fn 和 collect_fn）
                if isinstance(dispatch_mode, Dispatch):
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                #  获取执行函数（execute_fn）
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                wg_execute_fn_name = execute_mode["execute_fn_name"]
                try:
                    execute_fn = getattr(self, wg_execute_fn_name)
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception:
                    print(f"execute_fn {wg_execute_fn_name} is invalid")
                    raise

                # 生成并绑定新方法：
                # 利用func_generator将dispatch_fn、collect_fn组装到method_name上
                func = func_generator(
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                )
                try:
                    setattr(self, method_name, func)
                    method_names.append(method_name)
                except Exception as e:
                    raise ValueError(f"Fail to set method_name {method_name}") from e

        return method_names
```



一个栗子：

1. 自定义分发函数： **将 2 个输入参数扩展到所有 Worker**（`world_size` 个 Worker）

   * 例如，如果 `world_size=4`，输入 `x=[1, 2]` 会被扩展为 `x=[1, 2, 1, 2]`，使得每个 Worker 都能接收一个参数。

   ```python
   def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
       for arg in args:
           assert len(arg) == 2
           for i in range(worker_group.world_size - 2):
               arg.append(arg[i % 2])
       for k, v in kwargs.items():
           assert len(v) == 2
           for i in range(worker_group.world_size - 2):
               v.append(v[i % 2])
       return args, kwargs
   ```

2. `TestActor`（Worker 类）：

   ```python
   @ray.remote
   class TestActor(Worker):
       def __init__(self, x) -> None:
           super().__init__()
           self._x = x
   
       def foo(self, y):		# 普通方法：直接计算 self._x + y
           return self._x + y
   
       '''使用 @register 装饰器，指定分发模式 ALL_TO_ALL 和执行模式 RANK_ZERO：
       只会在 rank=0 的 Worker 上执行'''
       @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
       def foo_rank_zero(self, x, y):	
           return self._x + y + x
   
       '''使用自定义分发函数 two_to_all_dispatch_fn和收集函数 collect_all_to_all
       输入 x 和 y 会被 two_to_all_dispatch_fn 扩展后分发给所有 Worker'''
       @register(dispatch_mode={"dispatch_fn": two_to_all_dispatch_fn, "collect_fn": collect_all_to_all})
       def foo_custom(self, x, y):
           return self._x + y + x
   class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
   worker_group = RayWorkerGroup(resource_pool, class_with_args)
   output_ref = worker_group.foo_custom(x=[1, 2], y=[5, 6])
   '''每个 Worker 计算 self._x + y + x：
   Worker 0: 2 + 5 + 1 = 8
   Worker 1: 2 + 6 + 2 = 10
   Worker 2: 2 + 5 + 1 = 8
   Worker 3: 2 + 6 + 2 = 10'''
   assert output_ref == [8, 10, 8, 10]
   '''只有 rank=0 的 Worker 执行计算：2 + 2 + 1 = 5'''
   output_ref = worker_group.foo_rank_zero(x=1, y=2)
   assert output_ref == 5
   ```

   

## 参考

[Python 数据科学加速](https://scale-py.godaai.org/index.html)

[Ray Tutorial](https://scale-py.godaai.org/index.html)

[【AI Infra】【RLHF框架】一、VeRL中基于Ray的执行流程源码解析](https://zhuanlan.zhihu.com/p/29997527557)

[volcengine](https://github.com/volcengine)/[verl](https://github.com/volcengine/verl)

[tutorial.ipynb](https://github.com/volcengine/verl/blob/main/examples/ray/tutorial.ipynb)



