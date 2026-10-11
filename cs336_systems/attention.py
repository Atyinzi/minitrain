"""FlashAttention-2 implementations for CS336 Assignment 2.

Implemented by Codex for this workspace.  The PyTorch implementation mirrors
the tiled online-softmax algorithm.  The Triton implementation performs the
forward pass in one fused kernel; both classes use the same explicit,
recomputed backward equations so the math remains easy to inspect.
"""

from __future__ import annotations

import math

import torch

try:  # Triton is optional on CPU-only development machines.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None


def _causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Mask keys whose absolute position is after the query position."""
    nq, nk = scores.shape[-2:]
    q_idx = torch.arange(nq, device=scores.device)[:, None]
    k_idx = torch.arange(nk, device=scores.device)[None, :]
    return scores.masked_fill(k_idx > q_idx, -1e6)


def _attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equations 13-19 from the handout, recomputing P instead of saving it."""
    scale = q.shape[-1] ** -0.5
    # Explicit fp32 intermediates make the function valid for fp16/bf16 too;
    # mixed-dtype matmul would otherwise fail after subtracting fp32 L.
    qf, kf, vf = q.float(), k.float(), v.float()
    of, dof = output.float(), grad_output.float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if is_causal:
        scores = _causal_mask(scores)
    probabilities = torch.exp(scores - logsumexp.unsqueeze(-1))
    d = (of * dof).sum(dim=-1, keepdim=True)
    dp = torch.matmul(dof, vf.transpose(-2, -1))
    ds = probabilities * (dp - d)
    dq = (torch.matmul(ds, kf) * scale).to(q.dtype)
    dk = (torch.matmul(ds.transpose(-2, -1), qf) * scale).to(k.dtype)
    dv = torch.matmul(probabilities.transpose(-2, -1), dof).to(v.dtype)
    return dq, dk, dv


class FlashAttention2PyTorch(torch.autograd.Function):
    """Readable tiled FlashAttention-2 reference using standard PyTorch ops."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("expected q, k, v with shape (batch, sequence, d)")
        if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
            raise ValueError("incompatible q, k, v shapes")

        batch, n_queries, d = q.shape
        n_keys = k.shape[-2]
        q_tile_size = min(64, n_queries)
        k_tile_size = min(64, n_keys)
        scale = d**-0.5
        output = torch.empty_like(q)
        logsumexp = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)

        # Accumulators stay in fp32, matching the numerical recommendation in
        # the handout even when inputs use fp16/bf16.
        for q_start in range(0, n_queries, q_tile_size):
            q_end = min(q_start + q_tile_size, n_queries)
            q_tile = q[:, q_start:q_end].float()
            m = torch.full((batch, q_end - q_start), -torch.inf, device=q.device)
            ell = torch.zeros_like(m)
            acc = torch.zeros((batch, q_end - q_start, d), device=q.device)

            for k_start in range(0, n_keys, k_tile_size):
                k_end = min(k_start + k_tile_size, n_keys)
                scores = torch.matmul(q_tile, k[:, k_start:k_end].float().transpose(-2, -1)) * scale
                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=q.device)[:, None]
                    k_idx = torch.arange(k_start, k_end, device=q.device)[None, :]
                    scores = scores.masked_fill(k_idx > q_idx, -1e6)
                new_m = torch.maximum(m, scores.amax(dim=-1))
                correction = torch.exp(m - new_m)
                p_tilde = torch.exp(scores - new_m.unsqueeze(-1))
                ell = correction * ell + p_tilde.sum(dim=-1)
                acc = correction.unsqueeze(-1) * acc + torch.matmul(p_tilde, v[:, k_start:k_end].float())
                m = new_m

            output[:, q_start:q_end] = (acc / ell.unsqueeze(-1)).to(q.dtype)
            logsumexp[:, q_start:q_end] = m + torch.log(ell)

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logsumexp, q, k, v, output = ctx.saved_tensors
        dq, dk, dv = _attention_backward(q, k, v, output, grad_output, logsumexp, ctx.is_causal)
        return dq, dk, dv, None


if triton is not None:

    @triton.jit
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        n_queries,
        n_keys,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        q_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        q_offsets = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_offsets = tl.arange(0, K_TILE_SIZE)
        d_offsets = tl.arange(0, D)

        q = tl.load(
            q_ptr + batch_index * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd,
            mask=q_offsets[:, None] < n_queries,
            other=0.0,
        )
        m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
        ell = tl.zeros((Q_TILE_SIZE,), tl.float32)
        acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)

        for k_start in range(0, n_keys, K_TILE_SIZE):
            current_k = k_start + k_offsets
            k = tl.load(
                k_ptr + batch_index * stride_kb + current_k[:, None] * stride_kk + d_offsets[None, :] * stride_kd,
                mask=current_k[:, None] < n_keys,
                other=0.0,
            )
            v = tl.load(
                v_ptr + batch_index * stride_vb + current_k[:, None] * stride_vk + d_offsets[None, :] * stride_vd,
                mask=current_k[:, None] < n_keys,
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            valid = (q_offsets[:, None] < n_queries) & (current_k[None, :] < n_keys)
            if IS_CAUSAL:
                valid = valid & (q_offsets[:, None] >= current_k[None, :])
            scores = tl.where(valid, scores, -1.0e6)

            new_m = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m[:, None])
            ell = ell * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc=acc)
            m = new_m

        acc = acc / ell[:, None]
        tl.store(
            o_ptr + batch_index * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od,
            acc,
            mask=q_offsets[:, None] < n_queries,
        )
        tl.store(
            l_ptr + batch_index * stride_lb + q_offsets * stride_lq,
            m + tl.log(ell),
            mask=q_offsets < n_queries,
        )


class FlashAttention2Triton(torch.autograd.Function):
    """Fused Triton forward plus recomputed PyTorch backward."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        if triton is None:
            raise RuntimeError("Triton is required for FlashAttention2Triton")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("the Triton implementation requires CUDA tensors")
        if q.ndim != 3 or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError("expected compatible (batch, sequence, d) tensors")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        batch, n_queries, d = q.shape
        n_keys = k.shape[-2]
        if d > 128 or d < 16 or d & (d - 1):
            raise ValueError("head dimension must be a power of two in [16, 128]")

        output = torch.empty_like(q)
        logsumexp = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
        q_tile_size = 64
        k_tile_size = 64
        grid = (triton.cdiv(n_queries, q_tile_size), batch)
        _flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            *logsumexp.stride(),
            n_queries,
            n_keys,
            1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal,
            num_warps=4,
        )
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logsumexp, q, k, v, output = ctx.saved_tensors
        dq, dk, dv = _attention_backward(q, k, v, output, grad_output, logsumexp, ctx.is_causal)
        return dq, dk, dv, None

