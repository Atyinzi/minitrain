import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    pass

from .attention import FlashAttention2PyTorch, FlashAttention2Triton
from .distributed import DistributedDataParallel, FullyShardedDataParallel, ShardedOptimizer

__all__ = [
    "DistributedDataParallel",
    "FlashAttention2PyTorch",
    "FlashAttention2Triton",
    "FullyShardedDataParallel",
    "ShardedOptimizer",
]
