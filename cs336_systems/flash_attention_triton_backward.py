"""
Fused Triton FlashAttention-2 BACKWARD (handout Algorithm 2) — the optional
leaderboard kernel that replaces the PyTorch+torch.compile recompute backward.

Two kernels, so each program writes to disjoint outputs (no atomics):
  flash_bwd_dkdv_kernel : grid (Tk, B). One program per key tile; loops over
                          query tiles, accumulates dK(j), dV(j).
  flash_bwd_dq_kernel   : grid (Tq, B). One program per query tile; loops over
                          key tiles, accumulates dQ(i).
P is recomputed in both (cheap on-chip) rather than synchronized across blocks.

FlashAttentionTritonFull uses the existing forward kernel (flash_fwd_kernel)
for the forward and these kernels for the backward — fully fused, no recompute
in PyTorch. Causal masking + tile-skipping included.

Math (handout eq 13-19), scale = 1/sqrt(d):
  S  = Q Kᵀ * scale ;  P = exp(S - L) ;  D = rowsum(O ∘ dO)
  dV = Pᵀ dO ;  dP = dO Vᵀ ;  dS = P ∘ (dP - D)
  dQ = dS K * scale ;  dK = dSᵀ Q * scale
"""
import math

import torch
import triton
import triton.language as tl

from cs336_systems.flash_attention_triton import flash_fwd_kernel


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile = tl.program_id(0)
    b = tl.program_id(1)
    scale = scale.to(tl.float32)

    # this program owns key tile `key_tile`: load K_j, V_j once
    K_block_ptr = tl.make_block_ptr(
        K_ptr + b * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(key_tile * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(
        V_ptr + b * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(key_tile * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0))
    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)

    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    n_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    # causal: skip query tiles entirely before this key tile (fully masked)
    i_start = (key_tile * K_TILE_SIZE) // Q_TILE_SIZE if is_causal else 0

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + b * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(i_start * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0))
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + b * stride_dob, shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
        offsets=(i_start * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0))
    L_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(i_start * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,))
    D_block_ptr = tl.make_block_ptr(
        D_ptr + b * stride_Db, shape=(N_QUERIES,), strides=(stride_Dq,),
        offsets=(i_start * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,))

    key_pos = key_tile * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (Bk,)

    for i in range(i_start, n_q_tiles):
        Q_i = tl.load(Q_block_ptr)     # (Bq, D)
        dO_i = tl.load(dO_block_ptr)   # (Bq, D)
        L_i = tl.load(L_block_ptr)     # (Bq,)
        D_i = tl.load(D_block_ptr)     # (Bq,)

        S = (tl.dot(Q_i, tl.trans(K_j)) * scale).to(tl.float32)  # (Bq, Bk)
        if is_causal:
            q_pos = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            masked_value = tl.full((Q_TILE_SIZE, K_TILE_SIZE), -1.0e6, dtype=tl.float32)
            S = tl.where(key_pos[None, :] > q_pos[:, None], masked_value, S)
        P = tl.exp(S - L_i[:, None])                       # (Bq, Bk)

        dV += tl.dot(tl.trans(P).to(dO_i.dtype), dO_i)     # (Bk, D)
        dP = tl.dot(dO_i, tl.trans(V_j))                   # (Bq, Bk)
        dS = P * (dP - D_i[:, None])                       # (Bq, Bk)
        dK += tl.dot(tl.trans(dS).to(Q_i.dtype), Q_i) * scale  # (Bk, D)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + b * stride_dkb, shape=(N_KEYS, D), strides=(stride_dkk, stride_dkd),
        offsets=(key_tile * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0))
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + b * stride_dvb, shape=(N_KEYS, D), strides=(stride_dvk, stride_dvd),
        offsets=(key_tile * K_TILE_SIZE, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0))
    tl.store(dK_block_ptr, dK.to(dK_block_ptr.type.element_ty))
    tl.store(dV_block_ptr, dV.to(dV_block_ptr.type.element_ty))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile = tl.program_id(0)
    b = tl.program_id(1)
    scale = scale.to(tl.float32)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + b * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(query_tile * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0))
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + b * stride_dob, shape=(N_QUERIES, D), strides=(stride_doq, stride_dod),
        offsets=(query_tile * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0))
    L_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(query_tile * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,))
    D_block_ptr = tl.make_block_ptr(
        D_ptr + b * stride_Db, shape=(N_QUERIES,), strides=(stride_Dq,),
        offsets=(query_tile * Q_TILE_SIZE,), block_shape=(Q_TILE_SIZE,), order=(0,))

    Q_i = tl.load(Q_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    L_i = tl.load(L_block_ptr)
    D_i = tl.load(D_block_ptr)
    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_pos = query_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    # causal: only key tiles up to this query tile's last position contribute
    if is_causal:
        j_end = ((query_tile + 1) * Q_TILE_SIZE - 1) // K_TILE_SIZE + 1
    else:
        j_end = tl.cdiv(N_KEYS, K_TILE_SIZE)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + b * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(
        V_ptr + b * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0))

    for j in range(0, j_end):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        S = (tl.dot(Q_i, tl.trans(K_j)) * scale).to(tl.float32)
        if is_causal:
            key_pos = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            masked_value = tl.full((Q_TILE_SIZE, K_TILE_SIZE), -1.0e6, dtype=tl.float32)
            S = tl.where(key_pos[None, :] > q_pos[:, None], masked_value, S)
        P = tl.exp(S - L_i[:, None])
        dP = tl.dot(dO_i, tl.trans(V_j))
        dS = P * (dP - D_i[:, None])
        dQ += tl.dot(dS.to(K_j.dtype), K_j) * scale        # (Bq, D)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + b * stride_dqb, shape=(N_QUERIES, D), strides=(stride_dqq, stride_dqd),
        offsets=(query_tile * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0))
    tl.store(dQ_block_ptr, dQ.to(dQ_block_ptr.type.element_ty))


def flash_attention_triton_backward(Q, K, V, O, dO, L, is_causal, Bq=16, Bk=16):
    B, Nq, Dh = Q.shape
    Nk = K.shape[1]
    scale = 1.0 / math.sqrt(Dh)
    dO = dO.contiguous()

    Dvec = (O.to(torch.float32) * dO.to(torch.float32)).sum(-1)  # (B, Nq) fp32
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    flash_bwd_dkdv_kernel[(triton.cdiv(Nk, Bk), B)](
        Q, K, V, dO, L, Dvec, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        Dvec.stride(0), Dvec.stride(1),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        Nq, Nk, scale,
        D=Dh, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
    )
    flash_bwd_dq_kernel[(triton.cdiv(Nq, Bq), B)](
        Q, K, V, dO, L, Dvec, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        Dvec.stride(0), Dvec.stride(1),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        Nq, Nk, scale,
        D=Dh, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
    )
    return dQ, dK, dV


class FlashAttentionTritonFull(torch.autograd.Function):
    """Forward = your fused flash_fwd_kernel; backward = fused Triton (Algorithm 2)."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, Dh = Q.shape
        Nk = K.shape[1]
        Bq, Bk = 16, 16
        assert Nq % Bq == 0 and Nk % Bk == 0, "pick tile sizes that divide the seq lengths"
        O = torch.empty_like(Q)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(Dh)
        flash_fwd_kernel[(triton.cdiv(Nq, Bq), B)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, scale,
            D=Dh, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_attention_triton_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None
