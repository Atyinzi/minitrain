"""Distributed training components for CS336 Assignment 2.

Implemented by Codex for this workspace: overlapping DDP, optimizer-state
sharding, and an educational FSDP wrapper for the assignment's Linear and
Embedding layers.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer


def _dist_info() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


class _GradientBucket:
    """A flat communication buffer shared by several parameter gradients."""

    def __init__(self, parameters: list[nn.Parameter]):
        self.parameters = parameters
        self.offsets: dict[int, tuple[int, int]] = {}
        offset = 0
        for parameter in parameters:
            self.offsets[id(parameter)] = (offset, offset + parameter.numel())
            offset += parameter.numel()
        first = parameters[0]
        self.buffer = torch.empty(offset, device=first.device, dtype=first.dtype)
        self.ready: set[int] = set()
        self.handle: Any | None = None

    def copy_gradient(self, parameter: nn.Parameter) -> None:
        start, end = self.offsets[id(parameter)]
        self.buffer[start:end].copy_(parameter.grad.detach().reshape(-1))
        self.ready.add(id(parameter))

    def reset(self) -> None:
        self.ready.clear()
        self.handle = None


class DistributedDataParallel(nn.Module):
    """Educational DDP with asynchronous, bucketed gradient all-reduce."""

    def __init__(self, module: nn.Module, bucket_cap_mb: float = 25.0):
        super().__init__()
        if bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be greater than zero")
        self.module = module
        self.rank, self.world_size = _dist_info()
        self.bucket_cap_mb = bucket_cap_mb
        self.require_backward_grad_sync = True

        if self.world_size > 1:
            for tensor in self.module.state_dict().values():
                dist.broadcast(tensor, src=0)

        parameters = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        self._buckets = self._build_buckets(parameters)
        self._parameter_to_bucket = {
            id(parameter): bucket
            for bucket in self._buckets
            for parameter in bucket.parameters
        }
        for parameter in parameters:
            parameter.register_post_accumulate_grad_hook(self._make_grad_hook(parameter))

    def _build_buckets(self, parameters: list[nn.Parameter]) -> list[_GradientBucket]:
        """Group gradients without mixing devices or dtypes in one buffer."""
        cap_bytes = int(self.bucket_cap_mb * 1024 * 1024)
        buckets: list[_GradientBucket] = []
        current: list[nn.Parameter] = []
        current_bytes = 0
        current_key = None

        # Backward usually visits parameters in roughly reverse registration
        # order, so this order lets the first bucket launch early.
        for parameter in reversed(parameters):
            key = (parameter.device, parameter.dtype)
            parameter_bytes = parameter.numel() * parameter.element_size()
            if current and (key != current_key or current_bytes + parameter_bytes > cap_bytes):
                buckets.append(_GradientBucket(current))
                current = []
                current_bytes = 0
            current.append(parameter)
            current_bytes += parameter_bytes
            current_key = key
        if current:
            buckets.append(_GradientBucket(current))
        return buckets

    def _launch_bucket(self, bucket: _GradientBucket) -> None:
        if bucket.handle is not None:
            return
        bucket.buffer.div_(self.world_size)
        bucket.handle = dist.all_reduce(bucket.buffer, op=dist.ReduceOp.SUM, async_op=True)

    def _make_grad_hook(self, parameter: nn.Parameter):
        def hook(parameter: nn.Parameter) -> None:
            if parameter.grad is None or self.world_size == 1 or not self.require_backward_grad_sync:
                return
            bucket = self._parameter_to_bucket[id(parameter)]
            bucket.copy_gradient(parameter)
            if len(bucket.ready) == len(bucket.parameters):
                self._launch_bucket(bucket)

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    @contextmanager
    def no_sync(self):
        """Accumulate local gradients without launching all-reduce."""
        previous = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = previous

    def finish_gradient_synchronization(self) -> None:
        if self.world_size == 1:
            return

        # Normally every bucket was launched by its final gradient hook. This
        # fallback also handles parameters that did not participate in backward.
        for bucket in self._buckets:
            if bucket.handle is None:
                bucket.buffer.zero_()
                for parameter in bucket.parameters:
                    if parameter.grad is not None:
                        bucket.copy_gradient(parameter)
                self._launch_bucket(bucket)

        for bucket in self._buckets:
            bucket.handle.wait()
            for parameter in bucket.parameters:
                if parameter.grad is None:
                    continue
                start, end = bucket.offsets[id(parameter)]
                parameter.grad.copy_(bucket.buffer[start:end].view_as(parameter))
            bucket.reset()


class ShardedOptimizer(Optimizer):
    """ZeRO-1-like wrapper: shard optimizer state, then broadcast updates."""

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.rank, self.world_size = _dist_info()
        self.optimizer_cls = optimizer_cls
        self._inner_optimizer: Optimizer | None = None
        self._owners: dict[int, int] = {}
        self._owned_numel = [0] * self.world_size
        super().__init__(params, kwargs)

        local_groups = []
        for group in self.param_groups:
            local = [p for p in group["params"] if self._owners[id(p)] == self.rank]
            if local:
                local_groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": local})
        # Optimizer constructors reject an empty list. In realistic use every
        # rank owns parameters; the explicit check gives a useful error for tiny models.
        if not local_groups:
            raise ValueError("fewer optimizer parameters than distributed ranks")
        self._inner_optimizer = optimizer_cls(local_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for parameter in group["params"]:
            # Greedily assign the next tensor to the rank currently owning the
            # fewest elements. Balancing by tensor size matters for Transformer
            # embeddings and output projections, which are much larger than norms.
            owner = min(range(self.world_size), key=self._owned_numel.__getitem__)
            self._owners[id(parameter)] = owner
            self._owned_numel[owner] += parameter.numel()
        if self._inner_optimizer is not None:
            local = [p for p in group["params"] if self._owners[id(p)] == self.rank]
            if local:
                self._inner_optimizer.add_param_group(
                    {**{k: v for k, v in group.items() if k != "params"}, "params": local}
                )

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        # Schedulers update the public param_groups. Mirror values such as lr
        # and weight_decay into the rank-local optimizer before it steps.
        local_group_index = 0
        for group in self.param_groups:
            if not any(self._owners[id(p)] == self.rank for p in group["params"]):
                continue
            local_group = self._inner_optimizer.param_groups[local_group_index]
            for key, value in group.items():
                if key != "params":
                    local_group[key] = value
            local_group_index += 1
        loss = self._inner_optimizer.step(closure=closure, **kwargs)
        if self.world_size > 1:
            for group in self.param_groups:
                for parameter in group["params"]:
                    dist.broadcast(parameter.data, src=self._owners[id(parameter)])
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Clear every model gradient, not only those owned by this rank.
        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return only this rank's optimizer-state shard."""
        return self._inner_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the optimizer-state shard belonging to this rank."""
        self._inner_optimizer.load_state_dict(state_dict)


class FullyShardedDataParallel(nn.Module):
    """Parameter-sharded wrapper specialized to assignment Linear/Embedding.

    Master shards remain in fp32.  A full weight is gathered just in time for
    forward/backward, then its averaged gradient is reduce-scattered back to
    the local shard. Small parameters (for example RMSNorm scales) are kept
    replicated and use an all-reduce.
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        prefetch: bool = True,
    ):
        super().__init__()
        from cs336_basics.model import Embedding, Linear

        self.module = module
        self.compute_dtype = compute_dtype
        self.prefetch = prefetch
        self.rank, self.world_size = _dist_info()
        self._sharded: dict[int, dict[str, Any]] = {}
        self._forward_trace: list[nn.Module] = []
        self._forward_order_ready = False
        self._next_forward: dict[int, nn.Module] = {}
        self._next_backward: dict[int, nn.Module] = {}

        # Ranks may intentionally use different RNG seeds for data sampling.
        # Start every shard from rank zero's model before replacing full weights.
        if self.world_size > 1:
            for tensor in self.module.state_dict().values():
                dist.broadcast(tensor, src=0)

        for submodule in self.module.modules():
            if isinstance(submodule, (Linear, Embedding)):
                self._shard_weight(submodule)

        for parameter in self.module.parameters():
            if not parameter.requires_grad or id(parameter) in self._sharded:
                continue
            parameter.register_post_accumulate_grad_hook(self._replicated_grad_hook)
        self.module.register_forward_hook(self._model_forward_post_hook)

    def _shard_weight(self, submodule: nn.Module) -> None:
        from cs336_basics.model import Linear

        parameter = submodule.weight
        if id(parameter) in self._sharded:
            return
        flat = parameter.detach().reshape(-1)
        padding = (-flat.numel()) % self.world_size
        if padding:
            flat = torch.cat((flat, flat.new_zeros(padding)))
        shard_size = flat.numel() // self.world_size
        local = flat.narrow(0, self.rank * shard_size, shard_size).clone().float()
        metadata = {
            "parameter": parameter,
            "shape": tuple(parameter.shape),
            "numel": parameter.numel(),
            "shard_size": shard_size,
            "submodule": submodule,
            "is_linear": isinstance(submodule, Linear),
        }
        self._sharded[id(parameter)] = metadata
        parameter.data = local

        submodule.register_forward_pre_hook(self._forward_pre_hook)
        submodule.register_forward_hook(self._forward_post_hook)
        submodule.register_full_backward_pre_hook(self._backward_pre_hook)
        parameter.register_post_accumulate_grad_hook(self._sharded_grad_hook)

    def _finish_parameter_gradient(self, parameter: nn.Parameter) -> None:
        metadata = self._sharded[id(parameter)]
        handle = metadata.pop("pending_grad_handle", None)
        if handle is None:
            return
        handle.wait()
        local_grad = metadata.pop("pending_local_grad")
        local_grad.div_(self.world_size)
        previous_grad = metadata.pop("pending_previous_grad")
        if previous_grad is not None:
            local_grad.add_(previous_grad)
        parameter.grad = local_grad.to(parameter.data.dtype)
        metadata.pop("pending_full_grad")

    def _drain_completed_gradients(self) -> None:
        for metadata in self._sharded.values():
            handle = metadata.get("pending_grad_handle")
            if handle is not None and handle.is_completed():
                self._finish_parameter_gradient(metadata["parameter"])

    def _prefetch_weight(self, submodule: nn.Module, *, low_precision: bool) -> None:
        if not self.prefetch or self.world_size == 1:
            return
        parameter = submodule.weight
        metadata = self._sharded[id(parameter)]
        if "prefetch_handle" in metadata or "master_shard" in metadata:
            return
        comm_local = parameter.data.to(self.compute_dtype) if low_precision and self.compute_dtype is not None else parameter.data
        pieces = [torch.empty_like(comm_local) for _ in range(self.world_size)]
        metadata["prefetch_input"] = comm_local
        metadata["prefetch_pieces"] = pieces
        metadata["prefetch_handle"] = dist.all_gather(pieces, comm_local, async_op=True)

    def _gather_weight(self, submodule: nn.Module, *, low_precision: bool = True) -> None:
        parameter = submodule.weight
        metadata = self._sharded[id(parameter)]
        self._finish_parameter_gradient(parameter)
        local = parameter.data
        # Keep the exact fp32 storage alive. Reconstructing the master shard
        # from a gathered fp16 tensor would silently quantize optimizer state.
        metadata["master_shard"] = local
        # Keep a gradient accumulated by an earlier micro-batch while the
        # parameter temporarily changes from its local shard to a full weight.
        metadata["previous_grad"] = parameter.grad
        parameter.grad = None
        prefetched = metadata.pop("prefetch_handle", None)
        if prefetched is not None:
            prefetched.wait()
            full = torch.cat(metadata.pop("prefetch_pieces"))
            metadata.pop("prefetch_input")
        else:
            comm_local = local.to(self.compute_dtype) if low_precision and self.compute_dtype is not None else local
            if self.world_size == 1:
                full = comm_local
            else:
                pieces = [torch.empty_like(comm_local) for _ in range(self.world_size)]
                dist.all_gather(pieces, comm_local)
                full = torch.cat(pieces)
        parameter.data = full[: metadata["numel"]].view(metadata["shape"])

    def _restore_shard(self, submodule: nn.Module) -> None:
        parameter = submodule.weight
        metadata = self._sharded[id(parameter)]
        # Restore rather than recreate the master shard: the communicated full
        # weight may have been intentionally cast to fp16/bf16.
        parameter.data = metadata.pop("master_shard")
        parameter.grad = metadata.pop("previous_grad")

    def _forward_pre_hook(self, submodule: nn.Module, _inputs) -> None:
        if not self._forward_order_ready:
            self._forward_trace.append(submodule)
        self._gather_weight(submodule)
        next_submodule = self._next_forward.get(id(submodule))
        if next_submodule is not None:
            self._prefetch_weight(next_submodule, low_precision=True)

    def _forward_post_hook(self, submodule: nn.Module, _inputs, _output) -> None:
        self._restore_shard(submodule)

    def _backward_pre_hook(self, submodule: nn.Module, _grad_output) -> None:
        metadata = self._sharded[id(submodule.weight)]
        # Linear backward consumes the weight to compute grad_input, so it must
        # match the forward compute dtype. Embedding backward only accumulates
        # into the weight; keeping it fp32 matches the assignment baseline and
        # avoids an unnecessary second fp16 rounding.
        self._gather_weight(submodule, low_precision=metadata["is_linear"])
        submodule.weight.grad = None
        next_submodule = self._next_backward.get(id(submodule))
        if next_submodule is not None:
            next_metadata = self._sharded[id(next_submodule.weight)]
            self._prefetch_weight(next_submodule, low_precision=next_metadata["is_linear"])

    def _model_forward_post_hook(self, _module: nn.Module, _inputs, _output) -> None:
        if self._forward_order_ready:
            return
        for current, following in zip(self._forward_trace, self._forward_trace[1:]):
            self._next_forward[id(current)] = following
        backward_order = list(reversed(self._forward_trace))
        for current, following in zip(backward_order, backward_order[1:]):
            self._next_backward[id(current)] = following
        self._forward_order_ready = True

    def _sharded_grad_hook(self, parameter: nn.Parameter) -> None:
        if parameter.grad is None:
            return
        self._drain_completed_gradients()
        metadata = self._sharded[id(parameter)]
        grad = parameter.grad.detach().reshape(-1)
        padding = metadata["shard_size"] * self.world_size - grad.numel()
        if padding:
            grad = torch.cat((grad, grad.new_zeros(padding)))
        grad = grad.float().contiguous()
        if self.world_size == 1:
            local_grad = grad
            previous_grad = metadata.pop("previous_grad")
            parameter.data = metadata.pop("master_shard")
            if previous_grad is not None:
                local_grad.add_(previous_grad)
            parameter.grad = local_grad.to(parameter.data.dtype)
        else:
            local_grad = torch.empty(metadata["shard_size"], device=grad.device, dtype=grad.dtype)
            metadata["pending_full_grad"] = grad
            metadata["pending_local_grad"] = local_grad
            metadata["pending_previous_grad"] = metadata.pop("previous_grad")
            metadata["pending_grad_handle"] = dist.reduce_scatter_tensor(
                local_grad,
                grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            parameter.data = metadata.pop("master_shard")
            parameter.grad = None

    def _replicated_grad_hook(self, parameter: nn.Parameter) -> None:
        if parameter.grad is not None and self.world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(self.world_size)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for metadata in self._sharded.values():
            self._finish_parameter_gradient(metadata["parameter"])

    @torch.no_grad()
    def gather_full_parameters(self) -> dict[str, torch.Tensor]:
        result = {}
        for name, parameter in self.module.named_parameters():
            metadata = self._sharded.get(id(parameter))
            if metadata is None:
                result[name] = parameter.detach().clone()
                continue
            local = parameter.detach()
            if self.world_size == 1:
                full = local
            else:
                pieces = [torch.empty_like(local) for _ in range(self.world_size)]
                dist.all_gather(pieces, local)
                full = torch.cat(pieces)
            result[name] = full[: metadata["numel"]].view(metadata["shape"]).clone()
        return result

    @torch.no_grad()
    def gather_full_state_dict_rank0(self) -> dict[str, torch.Tensor] | None:
        """Reconstruct a regular model checkpoint only on rank zero.

        Every rank must call this method. Non-zero ranks send one parameter
        shard at a time instead of keeping another full model in memory.
        """
        result = None
        if self.rank == 0:
            result = {name: value.detach().clone() for name, value in self.module.state_dict().items()}

        for name, parameter in self.module.named_parameters():
            metadata = self._sharded.get(id(parameter))
            if metadata is None:
                if self.rank == 0:
                    result[name] = parameter.detach().clone()
                continue
            local = parameter.detach()
            if self.world_size == 1:
                full = local
            else:
                pieces = [torch.empty_like(local) for _ in range(self.world_size)] if self.rank == 0 else None
                dist.gather(local, gather_list=pieces, dst=0)
                if self.rank != 0:
                    continue
                full = torch.cat(pieces)
            result[name] = full[: metadata["numel"]].view(metadata["shape"]).clone()
        return result

    @torch.no_grad()
    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a regular full-model checkpoint into local parameter shards."""
        for name, parameter in self.module.named_parameters():
            source = state_dict[name].to(device=parameter.device, dtype=parameter.dtype)
            metadata = self._sharded.get(id(parameter))
            if metadata is None:
                parameter.copy_(source)
                continue
            flat = source.reshape(-1)
            padded_numel = metadata["shard_size"] * self.world_size
            if flat.numel() < padded_numel:
                flat = torch.cat((flat, flat.new_zeros(padded_numel - flat.numel())))
            start = self.rank * metadata["shard_size"]
            parameter.copy_(flat[start : start + metadata["shard_size"]])

        for name, buffer in self.module.named_buffers():
            if name in state_dict:
                buffer.copy_(state_dict[name].to(device=buffer.device, dtype=buffer.dtype))

