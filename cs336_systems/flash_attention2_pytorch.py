from __future__ import annotations

import math
import torch


class FlashAttention2ForwardPyTorch(torch.autograd.Function):
    """
    Pure PyTorch (no Triton) implementation of FlashAttention-2 forward pass
    following Algorithm 1 (online softmax across key tiles).

    Expected input shapes (common in the assignment):
      - Q, K, V: (batch, n_heads, seq_len, d_head)
    Also supports:
      - Q, K, V: (batch, seq_len, d)   (no head dim)
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        # (a) says: you can ignore is_causal for this task.
        # We still accept it to match the required signature.
        ctx.is_causal = is_causal

        assert Q.device.type == "cuda" or Q.device.type == "cpu", "Expected a torch Tensor"
        assert Q.dtype == K.dtype == V.dtype, "Q, K, V must have same dtype"
        assert Q.shape == K.shape == V.shape, "For this assignment setup we assume same shapes"

        # Support both (B, H, N, D) and (B, N, D)
        if Q.dim() == 4:
            B, H, N, D = Q.shape
            BH = B * H
            Q_ = Q.reshape(BH, N, D)
            K_ = K.reshape(BH, N, D)
            V_ = V.reshape(BH, N, D)
            out_shape = (B, H, N, D)
            L_shape = (B, H, N)
        elif Q.dim() == 3:
            B, N, D = Q.shape
            BH = B
            Q_ = Q
            K_ = K
            V_ = V
            out_shape = (B, N, D)
            L_shape = (B, N)
        else:
            raise ValueError(f"Unsupported Q/K/V rank: {Q.dim()}")

        # Tile sizes: must be at least 16x16 (problem statement).
        # You can tune later; for debugging, 32 is a reasonable default.
        Bq = 32
        Bk = 32
        assert Bq >= 16 and Bk >= 16

        scale = 1.0 / math.sqrt(D)
        ctx.sm_scale = scale

        # Allocate outputs (match input dtype for O; L is usually fp32 for stability)
        O = torch.empty((BH, N, D), device=Q_.device, dtype=Q_.dtype)
        L = torch.empty((BH, N), device=Q_.device, dtype=torch.float32)

        # Algorithm 1 over query tiles
        for bh in range(BH):
            # For each "batch-head" independently
            Q_bh = Q_[bh]  # (N, D)
            K_bh = K_[bh]  # (N, D)
            V_bh = V_[bh]  # (N, D)

            for q0 in range(0, N, Bq):
                Qi = Q_bh[q0:q0 + Bq, :]  # (Bq, D)

                # Initialize O^(0), l^(0), m^(0) in fp32 (as per tips/guidelines)
                m = torch.full((Qi.shape[0],), float("-inf"), device=Q_.device, dtype=torch.float32)
                l = torch.zeros((Qi.shape[0],), device=Q_.device, dtype=torch.float32)
                O_acc = torch.zeros((Qi.shape[0], D), device=Q_.device, dtype=torch.float32)

                # Single pass over key tiles j = 1..Tk
                for k0 in range(0, N, Bk):
                    Kj = K_bh[k0:k0 + Bk, :]  # (Bk, D)
                    Vj = V_bh[k0:k0 + Bk, :]  # (Bk, D)

                    # S_i^(j) = Qi Kj^T / sqrt(d)   -> shape (Bq, Bk)
                    # Accumulate in fp32 for stability
                    S = (Qi.float() @ Kj.float().transpose(0, 1)) * scale  # (Bq, Bk)

                    # m_i^(j) = max(m_i^(j-1), rowmax(S))
                    row_max = torch.max(S, dim=1).values  # (Bq,)
                    m_new = torch.maximum(m, row_max)     # (Bq,)

                    # P~ = exp(S - m_new)
                    P_tilde = torch.exp(S - m_new[:, None])  # (Bq, Bk) fp32

                    # l_i^(j) = exp(m_old - m_new)*l_old + rowsum(P~)
                    alpha = torch.exp(m - m_new)  # (Bq,)
                    l = alpha * l + torch.sum(P_tilde, dim=1)  # (Bq,)

                    # O_i^(j) = diag(alpha) O_old + P~ V
                    # Note: cast P~ to V dtype before matmul if you want to mimic kernel guideline
                    # but keep accumulation in fp32.
                    O_acc = alpha[:, None] * O_acc + (P_tilde.to(Vj.dtype).float() @ Vj.float())  # (Bq, D)

                    m = m_new

                # Final normalize: O = O_acc / l ; L = m + log(l)
                O_tile = (O_acc / l[:, None]).to(Q_.dtype)  # back to input dtype
                L_tile = m + torch.log(l)                   # fp32

                O[bh, q0:q0 + Qi.shape[0], :] = O_tile
                L[bh, q0:q0 + Qi.shape[0]] = L_tile

        # Save for backward pass (as required by the prompt)
        # They explicitly ask to save L, Q, K, V, O.
        if Q.dim() == 4:
            O_view = O.reshape(out_shape)
            L_view = L.reshape(L_shape)
        else:
            O_view = O.reshape(out_shape)
            L_view = L.reshape(L_shape)

        ctx.save_for_backward(L_view, Q, K, V, O_view)
        return O_view

    @staticmethod
    def backward(ctx, do):
        # 实现 backward，修复 NotImplementedError
        L, Q, K, V, O = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        # -----------------------------------------------------------------
        # 这里直接复制 _flash_attention_backward_logic 的逻辑即可
        # -----------------------------------------------------------------
        
        # 1. Recompute S
        S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
        if is_causal:
            seq_len = Q.shape[-2]
            mask = torch.triu(torch.ones((seq_len, seq_len), device=Q.device, dtype=torch.bool), diagonal=1)
            S = S.masked_fill(mask, float("-inf"))
        
        # 2. P = exp(S - L)
        P = torch.exp(S - L.unsqueeze(-1)).to(Q.dtype)

        # 3. D = rowsum(dO * O)
        D = torch.sum(do * O, dim=-1)

        # 4. dV = P^T dO
        dV = torch.matmul(P.transpose(-2, -1), do)

        # 5. dP = dO V^T
        dP = torch.matmul(do, V.transpose(-2, -1))

        # 6. dS = P * (dP - D)
        dS = P * (dP - D.unsqueeze(-1))

        # 7. dQ = dS K * scale
        dQ = torch.matmul(dS, K) * sm_scale

        # 8. dK = dS^T Q * scale
        dK = torch.matmul(dS.transpose(-2, -1), Q) * sm_scale

        return dQ, dK, dV, None