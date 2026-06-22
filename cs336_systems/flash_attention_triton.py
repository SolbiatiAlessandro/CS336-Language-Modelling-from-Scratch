import math
import torch
import triton
import triton.language as tl
from cs336_systems.flash_attention import _flash_backward_compiled

assert torch.cuda.is_available(), (
    "Triton needs a CUDA GPU. This will not run on Mac/MPS — use Modal or Colab GPU."
)
device = "cuda"
torch.manual_seed(0)
print("GPU:", torch.cuda.get_device_name(0))
print("triton:", triton.__version__)


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
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # ---- which query tile, which batch (plumbing) ----
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    scale = scale.to(tl.float32)

    # ---- block pointers: moving windows into global memory (plumbing) ----
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D), order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,), strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,), order=(0,),
    )

    # ---- load this query tile once; init running state (plumbing) ----
    Q = tl.load(Q_block_ptr)                                  # (Q_TILE_SIZE, D)
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)          # running numerator (your out_tile_accumulator)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)            # running denominator
    m = tl.full((Q_TILE_SIZE,), -1e6, dtype=tl.float32)       # running max

    # ---- stream key tiles: the single loop ----
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr)                              # (K_TILE_SIZE, D)
        V = tl.load(V_block_ptr)                              # (K_TILE_SIZE, D)

        # === TODO 1: scores  S = (Q @ K^T) * scale     -> (Q_TILE_SIZE, K_TILE_SIZE)
        #     ops: tl.dot, tl.trans
        S = (tl.dot(Q, tl.trans(K)) * scale).to(tl.float32)

        # === TODO (part c): if is_causal, mask future keys by adding -1e6
        #     q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)   # (Q_TILE,)
        #     k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)                  # (K_TILE,)
        #     mask future where k_idx > q_idx (use [:, None] / [None, :] + tl.where)
        #     leave this out for part (b); is_causal is a constexpr so the branch compiles away
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) 
            mask = k_idx.reshape(1, Q_TILE_SIZE) > q_idx.reshape(Q_TILE_SIZE, 1) 
            masked_value = tl.full((Q_TILE_SIZE, K_TILE_SIZE), -1.0e6, dtype=tl.float32)
            S = tl.where(mask, masked_value, S)

        # === TODO 2: new running max  m_new = max(m, rowmax(S))   -> (Q_TILE_SIZE,)
        #     ops: tl.maximum, tl.max(S, axis=-1)
        m_new = tl.maximum(m, tl.max(S, axis=-1))

        # === TODO 3: unnormalized probs  P = exp(S - m_new[:, None])   -> (Q_TILE, K_TILE)
        P = tl.exp(S - m_new.reshape((Q_TILE_SIZE, 1)))

        # === TODO 4: correction factor  alpha = exp(m - m_new)   -> (Q_TILE,)
        alpha = tl.exp(m - m_new)

        # === TODO 5: update denominator  l = alpha * l + rowsum(P)
        l = alpha * l + tl.sum(P, axis=-1)

        # === TODO 6: rescale + accumulate output  O = alpha[:, None] * O + P @ V
        #     cast P to V's dtype before the dot: tl.dot(P.to(V.dtype), V)
        O = alpha.reshape((Q_TILE_SIZE, 1)) * O + tl.dot(P.to(V.dtype), V)

        # === TODO 7: commit new max
        m = m_new

        # advance windows to the next key tile (plumbing)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # === TODO 8: finalize  O = O / l[:, None]   and   L = m + log(l)
    O = O / l.reshape(Q_TILE_SIZE, 1)
    L = m + tl.log(l)

    # ---- write results back (plumbing) ----
    tl.store(O_block_ptr, O.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        Nk = K.shape[1]
        # tile sizes: must be >= 16 and (for this kernel) divide the seq lengths.
        # the grader uses powers of 2 >= 16, so 16 is always safe; tune up for speed later.
        Bq, Bk = 16, 16
        assert Nq % Bq == 0 and Nk % Bk == 0, "pick tile sizes that divide the seq lengths"

        O = torch.empty_like(Q)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(D)
        grid = (triton.cdiv(Nq, Bq), B)   # (Tq, batch)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, scale,
            D=D, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        scale = 1.0 / (Q.shape[-1] ** 0.5)
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, ctx.is_causal, scale)
        return dQ, dK, dV, None
