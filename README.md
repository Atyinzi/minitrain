# MiniTrain

基于 Stanford CS336 Assignment 2 改造的轻量 Transformer 训练与推理实验项目。
保留可读的训练循环，支持训练恢复、验证、文本生成和分布式训练实验。

## 功能

- Transformer 语言模型、AdamW、梯度累积、余弦学习率与 warmup。
- BF16 混合精度、activation checkpointing、可选 `torch.compile`。
- naive / PyTorch SDPA / Triton attention 后端。
- DDP、ZeRO-1 优化器状态分片、实验性 FSDP 和预取。
- checkpoint 保存与恢复、JSONL 指标、验证集 loss。
- BPE tokenizer 加载、temperature / top-p 文本生成。
- attention、通信和并行策略的基准脚本及测试。

## 环境

依赖配置要求 Python 3.12 或 3.13，使用 uv 管理依赖：

```bash
uv sync --locked
```

GPU、Triton 和 NCCL 实验建议在 Linux / WSL2 + 兼容的 NVIDIA 驱动环境运行。
CPU 可以运行基础功能；GPU 和多卡功能需要相应硬件及运行环境。
`cs336-basics/` 是随仓库提供的本地依赖，安装时请保留该目录。

## 快速开始

在仓库根目录运行。下面使用随机 token 做训练流程检查，并非真实语料训练：

```bash
uv run python -m scripts.minitrain --device cpu --model-size tiny --steps 2 --context-length 16 --vocab-size 256
```

`tiny` 仅用于快速检查训练流程，不能代表真实模型质量。默认 `small` 有 12 层，
CPU 执行可能较慢。真实训练使用预先分词的连续
`uint16` token ID 文件，词表大小需与 tokenizer 一致：

```bash
uv run python -m scripts.minitrain --device cuda --train-bin data/train.bin --val-bin data/val.bin --vocab-size 10000 --steps 1000 --eval-every 100 --cosine-lr --mixed-precision
```

恢复训练：

```bash
uv run python -m scripts.minitrain --device cuda --train-bin data/train.bin --resume results/minitrain.pt --steps 2000
```

`--steps` 表示本次再训练的步数，上面的命令在原 checkpoint 的步数上再训练 2000 步。
恢复时请保持模型、数据和并行配置与原训练相容。ZeRO-1 的优化器分片文件也需要保留。

评估 checkpoint：

```bash
uv run python -m scripts.minitrain_inference --checkpoint results/minitrain.pt --val-bin data/val.bin --device cpu
```

文本生成需要与训练匹配的 tokenizer 文件：

```bash
uv run python -m scripts.minitrain_inference --checkpoint results/minitrain.pt --tokenizer data/tokenizer.pkl --prompt "Once upon a time" --max-new-tokens 100 --device cpu
```

查看完整参数：

```bash
uv run python -m scripts.minitrain --help
uv run python -m scripts.minitrain_inference --help
```

## 验证与实验

```bash
uv run pytest -q tests
```

其中 `test_minitrain_workflow.py` 使用合成 token 验证 CPU 训练、验证集评估、
checkpoint 保存与恢复的一致性，以及推理命令对同一 checkpoint 的评估。
GitHub Actions 在 Linux CPU 环境对完整测试集运行此命令；Triton 用例只有在
CUDA 与 Triton 都可用时才会运行。GPU 性能、多卡硬件和真实语料的模型质量
不在 CPU 测试的验证范围内。
其他 attention 和分布式测试位于 `tests/`，部分测试要求 CUDA 或多进程通信支持。
基准脚本位于 `scripts/benchmark_*.py`；中文实现说明见
[WRITEUP_GUIDE_CN.md](WRITEUP_GUIDE_CN.md)。其中历史测试记录和路径来自原开发环境，
不代表当前机器已复现全部 GPU 实验。

## 目录

```text
scripts/             训练、推理和基准入口
cs336_systems/       attention、分布式训练与 tokenizer
cs336-basics/        Transformer 基础模型及本地依赖配置
tests/               测试和小型测试夹具
```

训练数据、模型权重、运行结果、虚拟环境和缓存不随仓库发布。

## 来源与许可

本项目基于 Stanford CS336 Assignment 2 教学代码及本地改造，沿用已有包名以保持兼容。
原始 Stanford 版权声明和 MIT 许可保留在 [LICENSE](LICENSE)。
