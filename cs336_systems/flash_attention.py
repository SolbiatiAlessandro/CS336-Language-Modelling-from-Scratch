import numpy as np
import torch

def flash_forward_tiled(q, k, v, Bq=2, Bk=2, is_causal=False):    # TODO: implement the tiled online-softmax forward pass from Algorithm 1.
    # Return (o, L) with o.shape == q.shape and L.shape == q.shape[:-1].

    B, Nq, D = q.shape
    _, Nk, _ = k.shape
    Tq = int(Nq/Bq)
    Tk = int(Nk/Bk)
    print(f"B: {B}, Nq: {Nq}, Nk: {Nk}, D: {D}, Tq: {Tq}, Tk: {Tk}")
    Q_tiled = q.reshape(B, Tq, Bq, D)
    K_tiled = k.reshape(B, Tk, Bk, D)
    V_tiled = v.reshape(B, Tk, Bk, D)

    out = torch.zeros(B, Nq, D)
    logsumexp = torch.zeros(B, Nq, 1)
    
    for i in range(Tq):
        Q_tile = Q_tiled[:, i]
        out_tile_accumulator = torch.zeros((B, Bq, D))
        prev_tiled_softmax_denominator_cumsum = None
        prev_max = torch.ones(Bq, 1) * -1e6
        for k in range(Tk):
            K_tile, V_tile = K_tiled[:, k], V_tiled[:, k]
            self_attention_tile = Q_tile @ K_tile.transpose(-1, -2) / np.sqrt(D)
            max_tile, _ = self_attention_tile.max(axis=-1, keepdim=True)
            max_tile = torch.max(max_tile, prev_max)
            tiled_softmax_numerator = torch.exp(self_attention_tile - max_tile)
            scaling_factor = np.exp(prev_max - max_tile)
            if prev_tiled_softmax_denominator_cumsum is None:
                tiled_softmax_denominator_cumsum = tiled_softmax_numerator.sum(axis=-1, keepdims=True)
            else:
                scaled_cumsum = prev_tiled_softmax_denominator_cumsum * scaling_factor
                tiled_softmax_denominator_cumsum = scaled_cumsum + tiled_softmax_numerator.sum(axis=-1, keepdims=True)

            if k > 0:
                out_tile_accumulator = scaling_factor * out_tile_accumulator
     
            out_tile_accumulator += tiled_softmax_numerator @ V_tile

            prev_max = max_tile
            prev_tiled_softmax_denominator_cumsum = tiled_softmax_denominator_cumsum

        
        logsumexp[:, Bq*i:Bq*(i+1)] = max_tile + torch.log(tiled_softmax_denominator_cumsum)

        out_tile_accumulator = out_tile_accumulator / prev_tiled_softmax_denominator_cumsum
        out[:, Bq*i:Bq*(i+1), :] = out_tile_accumulator
    return out, logsumexp.reshape(B, Nq)

def _flash_backward(Q, K, V, O, dO, L, is_causal, scale):
    # === TODO 1: recompute scores  S = Q Kᵀ * scale          -> (B, Nq, Nk)
    B, T, C = Q.shape
    S = Q @ K.transpose(-2, -1) * scale 

    # === TODO (causal): if is_causal, mask future keys (k index > q index) to -1e6
    #     build qi = arange(Nq)[:, None], kj = arange(Nk)[None, :], use S.masked_fill(kj > qi, -1e6)
    #     (skip for the non-causal CPU test)
    if is_causal:
        qi = torch.arange(T, device=Q.device)[:, None]
        kj = torch.arange(T, device=Q.device)[None, :]
        S = S.masked_fill(kj > qi, -1e6) 

    # === TODO 2: reconstruct probs  P = exp(S - L[..., None])  -> (B, Nq, Nk)
    P = torch.exp(S - L.reshape(L.shape[0], L.shape[1], 1))

    # === TODO 3:           -> (B, Nq, 1)
    Di = (O * dO).sum(axis=-1, keepdims=True)

    # === TODO 4:                                  -> (B, Nk, D)
    dV = P.transpose(-1,-2) @ dO

    # === TODO 5:                                -> (B, Nq, Nk)
    dP = dO @ V.transpose(-1,-2)

    # === TODO 6:                          -> (B, Nq, Nk)
    dS = P * (dP - Di)

    # === TODO 7:                            -> (B, Nq, D)
    dQ = dS @ K * scale

    # S = Q @ K.T * scale
    # M = K.T
    # dM = Q.T @ dS * scale
    # dK = (dM).T = (Q.T @ dS).T = dS.T @ Q
    dK = dS.transpose(-1,-2) @ Q * scale 

    return dQ, dK, dV

# the torch.compile the handout asks for: compile the plain function, call it from backward
_flash_backward_compiled = torch.compile(_flash_backward)
    
class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = flash_forward_tiled(Q, K, V, is_causal=is_causal)
        ctx.save_for_backward(L,Q,K,V,O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        scale = 1.0 / (Q.shape[-1] ** 0.5)
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L, ctx.is_causal, scale)
        return dQ, dK, dV, None      # None = grad for is_causal

