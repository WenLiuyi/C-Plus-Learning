# verl框架：3. rollout流程解析

看一个TP的例子：**column-linear**

1. 运用**broadcast**操作，将输入矩阵复制到每个worker；
2. 将每个权重矩阵切分为若干个列，分别与输入矩阵相乘，最后通过**all-gather**操作结合。



![image-22](/Users/lisa/Desktop/image-22.png)

```bash
torchrun --nproc_per_node=2 column_linear.py
```

用以上`torchrun`命令启动以下代码：

```python
# column_linear.py
import torch
import torch.nn as nn

torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ['RANK']))

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 构建一个从column维度切分的linear layer
class ColumnTPLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size // int(os.environ['WORLD_SIZE']), bias=False).to(device='cuda')

    def forward(self, x):
      	# 1. 输入数据广播到所有GPU(输入数据形状为(10,2))
        ret = self.layer(x.to(device='cuda'))
        '''2. 局部矩阵乘法：
        Rank 0 计算：rank0_output = input_data @ rank0_weight.T  # shape=(10, 3)
        Rank 1 计算：rank1_output = input_data @ rank1_weight.T  # shape=(10, 3)
        ''' 
        '''2. all_gather合并计算结果：output_tensor 的形状是 (2, 10, 3)，其中：
				output_tensor[0] 是 Rank 0 的计算结果 (10, 3)
				output_tensor[1] 是 Rank 1 的计算结果 (10, 3)''' 
        output_tensor = torch.zeros(size=(int(os.environ['WORLD_SIZE']), ret.shape[0], ret.shape[1]), dtype=ret.dtype, device=ret.device)
        torch.distributed.all_gather_into_tensor(output_tensor, ret, async_op=False)
        # 4. 沿dim=-1（最后一个维度）拼接：shape=(10, 6)
        output_tensor = torch.cat(output_tensor.unbind(dim=0), dim=-1)	

        return output_tensor

    def load_weights(self, weight):
        rank = int(os.environ['RANK'])
				# 假设使用2个GPU（WORLD_SIZE=2），每个GPU负责6/2=3行的计算
        world_size = int(os.environ['WORLD_SIZE'])
        # Rank0获取前3行；Rank1获取后3行
        dim_per_rank = weight.shape[0] // world_size
        self.layer.weight.data.copy_(weight[rank*dim_per_rank: (rank+1)*dim_per_rank, :])

# 输入数据形状：(10,2)
batch_size = 10
input_data = torch.randn(batch_size, 2)

# init一个PyTorch的linear layer，并让我们构建的layer和它保持参数一致。
full_layer = torch.nn.Linear(2, 6, bias=False)
# 权重矩阵weight形状：(6,2)，6行对应output_size，2列对应input_size
weight = full_layer.state_dict()['weight']

tp_layer = ColumnTPLayer(2, 6)
tp_layer.load_weights(weight)

tp_ret = tp_layer(input_data).cpu()
fl_ret = full_layer(input_data).cpu()

torch.testing.assert_close(tp_ret, fl_ret)
```





`RayPPOTrainer`是运行在**单CPU/GPU节点**（默认CPU）上的驱动进程，负责协调PPO训练的三大核心功能：

1. **数据准备**：加载预处理后的parquet文件，输出按PPO mini-batch大小迭代的数据流

   ```py
   self.train_dataset = RLHFDataset(
       data_files=self.config.data.train_files,
       tokenizer=self.tokenizer,
       config=self.config.data
   )
   ```

2. **WorkerGroup初始化**：

   * **基础模式（角色分离）**：

     ```python
     '''FSDP后端：max_colocate_count设为1，合并所有WorkerGroup
     Megatron后端：max_colocate_count>1，对于不同模型使用不同的WorkerGroup
     '''
     resource_pool = RayResourcePool(
         process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
         use_gpu=True,
         max_colocate_count=1  # FSDP建议设为1，合并所有WorkerGroup
     )
     actor_rollout_worker_group = MegatronRayWorkerGroup(
         resource_pool=resource_pool,
         ray_cls_with_init=actor_rollout_cls,
         default_megatron_kwargs=config.actor_rollout.megatron
     )
     ```

     



初始化`RayPPOTrainer`时，调用`init_workers`方法，包含以下语句：

```python
self.actor_rollout_wg = all_wg["actor_rollout"]
self.actor_rollout_wg.init_model()
```



从`RayPPOTrainer`的`fit`函数进入：

```python
# generate a batch
for batch_dict in self.train_dataloader:
  with _timer("gen", timing_raw):		# 性能监控：记录轨迹生成耗时
      if not self.async_rollout_mode:	# 同步模式
          gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
      else:														# 异步模式
          self.async_rollout_manager.wake_up()
          gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
          self.async_rollout_manager.sleep()

  # 采用REMAX优势估计时：通过对比采样轨迹（带探索）和基线轨迹（确定性）的奖励差异，优化优势估计的方差。
  if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
      with _timer("gen_max", timing_raw):
          gen_baseline_batch = deepcopy(gen_batch)
          # 禁用采样，使用贪婪解码：强制模型选择最高概率的token（确定性输出）
          gen_baseline_batch.meta_info["do_sample"] = False	
          gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

          batch = batch.union(gen_baseline_output)	# 合并基线数据到主batch
          reward_baseline_tensor = self.reward_fn(batch)	# 计算基线奖励
          reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1) # 汇总序列级奖励

          batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

          batch.batch["reward_baselines"] = reward_baseline_tensor	# 存储基线奖励
					# 资源清理
          del gen_baseline_batch, gen_baseline_output
```

