from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,   # <<< 新增：必须 tl.constexpr
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # ---- Block pointers (Q fixed per program; K/V advanced in the loop) ----
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # ---- Load Q tile once ----
    Qi = tl.load(Q_block_ptr)  # (Bq, D)

    # ---- On-chip buffers in fp32 (per assignment tips) ----
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)   # running max
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)           # running denom proxy
    O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)     # running output numerator

    # ---- Single loop over key tiles ----
    n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(n_key_tiles):
        k_tile_index = j
        Kj = tl.load(K_block_ptr)  # (Bk, D)
        Vj = tl.load(V_block_ptr)  # (Bk, D)

        # S = Qi @ Kj^T * scale   -> (Bq, Bk)
        # Use fp32 accumulation for stability
        S = tl.dot(Qi, tl.trans(Kj)).to(tl.float32) * scale

        if is_causal:
            # 构造当前 tile 的 query/key 全局索引
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)   # (Bq,)
            k_idx = k_tile_index   * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)     # (Bk,)
            # 下三角：允许 k <= q
            mask = q_idx[:, None] >= k_idx[None, :]                               # (Bq, Bk)
            # masked 的位置加 -1e6
            S = tl.where(mask, S, S + (-1.0e6))

        # rowmax over keys
        row_max = tl.max(S, axis=1)              # (Bq,)
        m_new = tl.maximum(m, row_max)           # (Bq,)

        # P~ = exp(S - m_new)
        P_tilde = tl.exp(S - m_new[:, None])     # (Bq, Bk), fp32

        # l = exp(m_old - m_new) * l + rowsum(P~)
        alpha = tl.exp(m - m_new)                # (Bq,)
        l = alpha * l + tl.sum(P_tilde, axis=1)

        # O_acc = diag(alpha) O_acc + P~ V
        # Cast P~ to V dtype before dot; accumulate in fp32 via acc
        P_cast = P_tilde.to(Vj.dtype)
        O_acc = alpha[:, None] * O_acc
        O_acc = tl.dot(P_cast, Vj, acc=O_acc)

        m = m_new

        # Advance K/V pointers to next key tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # ---- Final normalize and write ----
    O_out = (O_acc / l[:, None]).to(O_block_ptr.type.element_ty)
    L_out = (m + tl.log(l)).to(tl.float32)  # we store L as fp32

    tl.store(O_block_ptr, O_out)
    tl.store(L_block_ptr, L_out)


class FlashAttention2ForwardTriton(torch.autograd.Function):
    """
    Part (b): forward uses Triton kernel; backward暂时NotImplemented（题面(b)没要求实现 backward）。
    Supports Q/K/V shaped (B, N, D) or (B, H, N, D) by folding (B*H) into batch dimension.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        # (b) doesn't require causal; (c) will add it. Keep arg for API compatibility.
        ctx.is_causal = is_causal

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.dtype == K.dtype == V.dtype
        assert Q.shape == K.shape == V.shape

        # Fold head into batch if needed: (B,H,N,D) -> (BH,N,D)
        if Q.dim() == 4:
            B, H, N, D = Q.shape
            BH = B * H
            Q_ = Q.reshape(BH, N, D)
            K_ = K.reshape(BH, N, D)
            V_ = V.reshape(BH, N, D)
            out_shape = (B, H, N, D)
            L_shape = (B, H, N)
        elif Q.dim() == 3:
            BH, N, D = Q.shape[0], Q.shape[1], Q.shape[2]
            Q_, K_, V_ = Q, K, V
            out_shape = (BH, N, D)
            L_shape = (BH, N)
        else:
            raise ValueError(f"Unsupported rank: {Q.dim()}")

        # Heuristic tile sizes (must be >=16; N is power-of-2 >=16 per prompt)
        if N >= 32:
            Q_TILE_SIZE = 32
            K_TILE_SIZE = 32
        else:
            Q_TILE_SIZE = 16
            K_TILE_SIZE = 16

        scale = 1.0 / math.sqrt(D)

        O = torch.empty_like(Q_)
        L = torch.empty((BH, N), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(N, Q_TILE_SIZE), BH)

        flash_fwd_kernel[grid](
            Q_, K_, V_,
            O, L,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N, N_KEYS=N,
            scale=scale,
            is_causal=is_causal,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            num_warps=4,
        )

        # Reshape back
        O_view = O.reshape(out_shape)
        L_view = L.reshape(L_shape)

        # Save for backward (as required in (a); keep consistent)
        ctx.save_for_backward(L_view, Q, K, V, O_view)
        return O_view

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Part (b) only: backward will be implemented in flash_backward.")
