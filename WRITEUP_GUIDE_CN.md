# CS336 Assignment 2 中文实现与实验指南

> 本文件由 Codex 为本工作区实现。目标不是只给出答案，而是让你能沿着
> “正确性测试 -> baseline -> profiler -> 优化 -> 再测量 -> 解释”复现实验。

## 0. 当前完成度与文件归属

以下是本次新增或修改、由 Codex 完成的文件：

| 文件 | 作用 | 验证状态 |
|---|---|---|
| `cs336_systems/attention.py` | tiled PyTorch FlashAttention-2、Triton forward、重计算 backward | CPU PyTorch 测试通过；Triton 需可用 CUDA 驱动复测 |
| `cs336_systems/distributed.py` | overlap DDP、optimizer-state sharding、教学版 FSDP | 公开 CPU/Gloo 测试通过 |
| `cs336_systems/__init__.py` | 导出上述实现 | 静态编译通过 |
| `tests/adapters.py` | 将原始公开测试接到实现 | 公开测试已运行 |
| `scripts/benchmark_transformer.py` | 端到端时间、mixed precision、compile、NVTX、memory snapshot | 静态编译通过；数值需 GPU |
| `scripts/benchmark_attention.py` | 原生/compile/Triton attention 扫描 | 静态编译通过；数值需 GPU |
| `scripts/benchmark_collectives.py` | 多 GPU all-reduce benchmark | 静态编译通过；数值需多 GPU |
| `scripts/benchmark_parallel.py` | naïve/flat/overlap DDP、sharded optimizer、FSDP 对比 | 静态编译通过；数值需多 GPU |
| `scripts/profile_checkpointing.py` | 验证 checkpoint segment size | 静态编译通过；数值需 GPU |

没有伪造 B200、Nsight 或显存实验数据。当前机器的 NVIDIA 驱动版本低于本项目
PyTorch 所需版本，因此所有要求截图、B200 latency 或多 GPU NCCL 数字的问题，都应在
驱动兼容的 CUDA 机器上运行下述命令后填写。代码实现与不依赖 GPU 的正确性测试已经完成。

## 1. 建议学习顺序

1. 先读 `scripts/benchmark_transformer.py`，理解同步、warmup 和分阶段计时。
2. 再读 `cs336_systems/attention.py` 中的 PyTorch tiled 实现，对照 online softmax 三个状态：`m`、`ell`、`acc`。
3. 再读同一文件的 Triton kernel，观察一个 program 如何负责一个 query tile。
4. 然后读 `DistributedDataParallel`：梯度一 ready 就发起异步 all-reduce。
5. 读 `ShardedOptimizer`：参数完整复制，只有 optimizer state 分片。
6. 最后读 `FullyShardedDataParallel`：master weight 是 FP32 shard，计算前临时 all-gather。

## 2. 正确性测试

```bash
uv run pytest tests/test_attention.py -q
uv run pytest tests/test_ddp.py -q
uv run pytest tests/test_sharded_optimizer.py -q
uv run pytest tests/test_fsdp.py -q
```

本机通过 WSL 实测：PyTorch attention、DDP、sharded optimizer、FSDP FP32/FP16 均通过。
Triton 的四项测试被跳过，原因不是代码断言失败，而是本机驱动只支持 CUDA 12.3，当前
PyTorch 构建要求更新的驱动。

## 3. Benchmark：如何避免测错

### 3.1 为什么必须 warm up 和 synchronize

CUDA kernel launch 是异步的。若不调用 `torch.cuda.synchronize()`，Python 计时主要测到
CPU 提交 kernel 的时间，而不是 GPU 真正执行时间。首次运行还包括 CUDA context 初始化、
allocator 建池、cuBLAS heuristic 选择、Triton/torch.compile 编译，因此 0 次 warmup 往往
显著偏慢；1-2 次仍可能没有覆盖全部动态 shape 或编译路径，5 次通常更稳定。

运行示例：

```bash
uv run python scripts/benchmark_transformer.py --model-size small --mode forward
uv run python scripts/benchmark_transformer.py --model-size small --mode backward
uv run python scripts/benchmark_transformer.py --model-size small --mode train
uv run python scripts/benchmark_transformer.py --model-size small --mode train --mixed-precision
uv run python scripts/benchmark_transformer.py --model-size small --mode train --compile
```

输出同时包含 mean、sample standard deviation、peak allocated 和 peak reserved。比较优化前后时，
固定 GPU、PyTorch 版本、batch、context、dtype、warmup 和 measurement steps。

### 3.2 mixed-precision 的 dtype 答案

CUDA FP16 autocast 中：model parameters 仍为 FP32；`fc1` 输出 FP16；LayerNorm 的敏感
reduction 以 FP32 执行并输出 FP32；`fc2` logits 为 FP16；cross-entropy loss 为 FP32；
累积到 FP32 parameters 上的 gradients 为 FP32。LayerNorm 的均值、方差和平方和容易出现
overflow/underflow 或消减误差；BF16 虽有 FP32 相同的指数范围，但 mantissa 更短，所以
reduction 仍适合保留 FP32。

`0.01` 累加 1000 次时，FP32 约为 `10.00013`；FP16 accumulator 约为 `9.953125`，因为每次
累加都舍入；将 FP16 的 `0.01` 累加进 FP32 时约为 `10.00214`，误差只来自常数先量化成
FP16，而不是 accumulator 每一步持续损失精度。

## 4. Profiler 与瓶颈分析

### 4.1 Nsight 命令

脚本中的 measurement 区间已经用 NVTX 标注：

```bash
uv run nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  --capture-range=nvtx --capture-range-end=stop \
  --output results/small_ctx512 \
  -- python scripts/benchmark_transformer.py --model-size small --context-length 512 --mode train
```

需要记录而不是猜测的内容：forward 总时长、累计时间最大的 kernel、调用次数、matmul 占比、
softmax/RMSNorm/elementwise kernel 占比。常见但必须由 trace 验证的趋势是：小矩阵和短序列
更受 launch/内存带宽限制；模型与序列变大后 GEMM 占比通常上升；optimizer step 包含大量
逐元素读写，会降低整步中 matmul 的时间占比。

### 4.2 三个算子的预期瓶颈

- Attention：朴素实现显式写入 `[B, N, N]` scores/probabilities，长序列时 HBM 流量和
  二次方 residual memory 是核心瓶颈。
- RMSNorm：FLOPs 很少，却要多次读取 activation 并做 reduction，典型 memory-bound；fusion
  减少中间 tensor 和 kernel launches。
- SwiGLU/MLP：三个大矩阵乘法主导 FLOPs，尺寸足够大时更接近 compute-bound；SiLU 和乘法
  是可融合的 bandwidth/launch overhead。

## 5. Memory profiling 与 activation checkpointing

```bash
uv run python scripts/benchmark_transformer.py \
  --model-size xl --context-length 2048 --mode train --steps 1 \
  --memory-snapshot results/xl_ctx2048_train.pickle
```

把 pickle 拖入 `https://pytorch.org/memory_viz`，分别保存 forward-only 与 train 的 Active
Memory Timeline。xl residual-stream activation（batch 4、context 2048、d_model 2560、FP32）为：

```text
4 * 2048 * 2560 * 4 bytes / 1024^2 = 80 MiB
```

无 checkpoint 时 N 个 block 同时保存 residual，峰值为 O(N)。只允许一层 checkpoint 时，
每段 k 层，需要长期保存约 N/k 个 segment inputs，重计算时同时物化约 k 层 residual，峰值
近似 `O(N/k + k)`；令两项平衡得到 `k ≈ sqrt(N)`。xl 有 32 层，所以从 6 附近开始，并测
相邻的 4、6、8：

```bash
uv run python scripts/profile_checkpointing.py --segment-sizes 4 6 8
```

若忽略计算成本并允许递归嵌套，采用二分 recursive checkpoint：对区间取中点 checkpoint，
backward 时递归重算左右半区。峰值 activation memory 为 O(log N)，总 compute 为
O(N log N)。它比单层的 O(sqrt(N)) memory 更低，但重计算显著增加。

## 6. FlashAttention-2：代码应该怎样读

朴素 attention 保存 N x N 的 P；FlashAttention 只保留当前 Q/K tile，并为每一行保存：

- `m`：目前见过的最大 score；
- `ell`：在当前最大值基准下的指数和；
- `acc`：在相同基准下的未归一化 `P @ V`。

当新 tile 带来更大的 maximum 时，用 `alpha = exp(old_m - new_m)` 重缩放旧的 `ell` 和
`acc`。最终 `O = acc / ell`，`L = m + log(ell)`。因此 HBM 上不再出现完整 N x N matrix，
saved memory 从 O(N^2) 降到 O(Nd)。backward 用 `P = exp(S-L)` 重计算 probabilities；
`D = rowsum(O * dO)` 后有 `dS = P * (dP-D)`。

benchmark：

```bash
uv run python scripts/benchmark_attention.py --batch-size 1 \
  --sequence-lengths 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
  --dimensions 16 32 64 128 --dtype bfloat16
```

为什么小输入可能变慢：tile setup、JIT 与 kernel launch 是固定成本，且小矩阵无法提供足够
并行度。为什么长序列变快且更省内存：省掉 N x N 的 HBM materialization 后，online
softmax 的额外 FLOPs 通常比全局内存读写便宜。

## 7. DDP 与 collective communication

### 7.1 all-reduce baseline

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_collectives.py
torchrun --standalone --nproc-per-node=4 scripts/benchmark_collectives.py
torchrun --standalone --nproc-per-node=6 scripts/benchmark_collectives.py
```

小 tensor 常由 latency 主导，无法达到链路峰值带宽；tensor 增大后 effective bandwidth 上升。
ring all-reduce 的理想流量因子为 `2(N-1)/N`，GPU 数增加会增加步骤数，但每步 chunk 变小。

题目中的 alternate ring 每一步发送完整 S bytes，共 N-1 步，所以时间为：

```text
T = (N - 1) S / W
```

它比 reduce-scatter + all-gather 的 `2(N-1)S/(NW)` 差约 N/2 倍。

### 7.2 三种 DDP

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_parallel.py --strategy naive
torchrun --standalone --nproc-per-node=2 scripts/benchmark_parallel.py --strategy flat
torchrun --standalone --nproc-per-node=2 scripts/benchmark_parallel.py --strategy overlap
```

- naïve：backward 完成后逐参数 all-reduce；调用多，且通信全暴露在 critical path。
- flat：一次大 all-reduce，amortize launch latency，但仍完全暴露。
- overlap：`register_post_accumulate_grad_hook` 在每个 gradient ready 时发异步 all-reduce，
  尽量把通信藏在剩余 backward compute 下；optimizer 前统一 wait。

## 8. Optimizer-state sharding 与 FSDP

FP32 AdamW 近似内存（P 为参数个数）：parameters `4P`、gradients `4P`、m/v `8P`，合计
`16P` bytes/rank。只 shard optimizer state 后为 `(8 + 8/N)P`；FSDP 再 shard parameters 和
gradients，忽略临时 all-gather buffer 时约为 `16P/N`。

本实现的 optimizer sharding 是教学版 ZeRO-1：每个 rank 仅更新自己的参数子集并逐参数
broadcast。ZeRO-1 的核心内存结论相同，但生产实现通常按 flat bucket 分片并用高吞吐
collective，减少 Python/collective launch overhead；本实现仍在每个 rank 保存完整参数和梯度。

FSDP 的关键 invariant：FP32 master shard 是 optimizer 的真值来源。forward/backward 前可以
先把 shard cast 成 BF16/FP16 再 all-gather，以减少一半通信量；计算结束必须恢复原 FP32
master storage，不能从低精度 full weight 切片重建，否则每步会永久量化权重。

## 9. 并行策略数学推导

记 batch B、model width D、FFN width Dff、设备算力 C FLOP/s、单设备出口带宽 W byte/s，
FP16 每元素 2 bytes。

### 9.1 Data Parallel

FFN backward 有 6 个同规模 matmul，因此每设备：

```text
F_backward = 12 B D Dff / N_DP
T_comm = 12 (N_DP - 1) D Dff / (N_DP W)
```

要求通信不慢于计算，得到：

```text
N_DP <= 1 + B W / C
```

### 9.2 FSDP

```text
F_forward  =  6 B D Dff / N_FSDP
F_backward = 12 B D Dff / N_FSDP
T_forward_comm  =  6 (N_FSDP - 1) D Dff / (N_FSDP W)
T_backward_comm = 12 (N_FSDP - 1) D Dff / (N_FSDP W)
```

forward 是一次 weight all-gather，backward 是 all-gather + reduce-scatter。两者分别比较后都得：

```text
N_FSDP <= 1 + B W / C
```

### 9.3 Tensor Parallel

设每个 rank 已保存 `X, X1_i, X2_i, Z_i`，完整 backward 方程为：

```text
dZ_i  = dY W3_i^T
dX2_i = dZ_i * f(X1_i)
dX1_i = dZ_i * f'(X1_i) * X2_i
dX_partial_i = dX1_i W1_i^T + dX2_i W2_i^T
dX = all-reduce({dX_partial_i})
dW3_i = Z_i^T dY
dW2_i = X^T dX2_i
dW1_i = X^T dX1_i
```

也就是说，每个 rank 先算局部 `dZ, dX1, dX2, dW1, dW2, dW3`，然后将局部
`dX_partial` 做 all-reduce。forward 与 backward 各通信一次 BD activation：

```text
F_forward  =  6 B D Dff / N_TP
F_backward = 12 B D Dff / N_TP
T_forward_comm = T_backward_comm = 4 (N_TP - 1) B D / (N_TP W)

forward:  N_TP <= 1 + (3/2) Dff W / C
backward: N_TP <= 1 + 3 Dff W / C
```

### 9.4 2D FSDP + TP

```text
F_forward = 6 B D Dff / (N_FSDP N_TP)

T_FSDP = 6 (N_FSDP - 1) D Dff / (N_FSDP N_TP W)
T_TP   = 4 (N_TP - 1) B D / (N_FSDP N_TP W)
```

两轴通信可重叠时 `T_comm = max(T_FSDP, T_TP)`，因此最优可扩展设备数满足：

```text
N = N_FSDP N_TP
N <= (1 + B W/C) (1 + (3/2) Dff W/C)
```

两轴不能重叠时，令 `a = BW/C`、`c = 2B/(3Dff)`。约束为
`(N_FSDP-1) + c(N_TP-1) <= a`。连续最优点满足 `N_FSDP = c N_TP`，因此：

```text
N <= (a + 1 + c)^2 / (4c)
```

实际部署还要把两个维度取合法整数，并服从节点拓扑；通常把通信更频繁的 TP 放在节点内高速
互联，把 FSDP 放到较慢的跨节点轴。

## 10. 最终实验记录模板

每张表至少保存这些 metadata：GPU 型号、driver、CUDA、PyTorch、Triton、dtype、batch、
context、warmup、repetitions。对每项优化写四句话：baseline 是什么；profiler 说明哪里慢；
优化改变了计算/通信/内存中的哪一项；哪些 shape 更快、哪些更慢以及原因。

不要把 “OOM” 删除，它本身就是结果；不要把 compiled 第一次运行混入 steady-state latency；
不要跨不同机器直接比较绝对时间。
