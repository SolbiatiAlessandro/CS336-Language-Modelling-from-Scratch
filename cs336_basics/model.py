import torch
import torch.nn as nn
import math
from collections.abc import Callable, Iterable
from typing import Optional
import functools


_FLASH_ATTENTION_TRITON = None


def _get_flash_attention_triton():
    global _FLASH_ATTENTION_TRITON
    if _FLASH_ATTENTION_TRITON is None:
        from cs336_systems.flash_attention_triton_backward import FlashAttentionTritonFull

        _FLASH_ATTENTION_TRITON = FlashAttentionTritonFull
    return _FLASH_ATTENTION_TRITON


def flash_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    B, H, T, D = q.shape
    flash_attention = _get_flash_attention_triton()
    q = q.contiguous().reshape(B * H, T, D)
    k = k.contiguous().reshape(B * H, T, D)
    v = v.contiguous().reshape(B * H, T, D)
    output = flash_attention.apply(q, k, v, is_causal)
    return output.reshape(B, H, T, D)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, debug=True, optional_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        std = math.sqrt(2 / (in_features + out_features))
        if optional_weight is None:
            self.W = nn.Parameter(
                nn.init.trunc_normal_(
                    torch.zeros(
                        out_features,
                        in_features,
                        device=device,
                        dtype=dtype,
                    ),
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                ),
            )
        else:
            self.W = optional_weight

        # self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T  # + self.b


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(
                    num_embeddings,
                    embedding_dim,
                    device=device,
                    dtype=dtype,
                ),
                std=std,
                a=-3 * std,
                b=3 * std,
            ),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    # TODO: try performance with layer norm as well
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.gain = nn.Parameter(torch.ones(d_model,device=device))
        self.e = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.e + ((x**2).sum(axis=-1, keepdims=True) / self.d_model))
        result = (x / rms) * self.gain
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    # TODO: try performance with ReLU as well
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if d_ff is None:
            self.d_ff = round((8 / 3) * d_model / 64) * 64
        self.linear1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.linear3 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.linear1(x)
        y1 = y1 * torch.sigmoid(y1)
        return self.linear3(y1 * self.linear2(x))

rope_cache = {} # (T,C,theta) -> R

class RoPE(nn.Module):
    #TODO optimize to not have the full R matrix
    #TODO try to train with and wihtout rope with normal positional embedding
    #It constructs [T, C, C], costing O(TC²) memory instead of O(TC).
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        T,C = max_seq_len, d_k
        self.T = T
        self.C = C
        self.theta = theta
        rope_cache_key = (T,C,theta)
        if rope_cache_key not in rope_cache.keys():
            R = torch.zeros((T,C,C),device=device)
            for i in range(T):
                for half_k in range(0, int(C/2)):
                    theta = self._get_theta(i, half_k, d_k)
                    k = half_k * 2
                    c = torch.tensor(math.cos(theta),device=device)
                    s = torch.tensor(math.sin(theta),device=device)
                    R[i, k, k] = c
                    R[i, k, k+1] = -1 * s
                    R[i, k+1, k] = s
                    R[i, k+1, k+1] = c
            rope_cache[rope_cache_key] = R
        self.register_buffer("R", rope_cache[rope_cache_key], persistent=False)

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]
        R = self.R[:seq_len, :, :]
        if token_positions is not None:
            raise NotImplementedError("token_positionsin RoPE is for KV cache, not supported")
        y = R @ x.reshape(list(x.shape) + [1])
        return y.reshape(x.shape)

    def _get_theta(self, i, half_k, d_k):
        THETA = self.theta
        return i / math.pow(THETA, (2*half_k)/ d_k)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.max(dim=-1, keepdims=True).values
    return torch.exp(x - m)/torch.exp(x - m).sum(axis=dim, keepdims=True)


def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
    self_attention = (q @ k.transpose(-1, -2))/math.sqrt(q.shape[-1])
    self_attention = self_attention * mask + (~mask) * -1e6
    probs = softmax(self_attention, dim=-1)
    return probs @ v

class  multihead_self_attention_with_rope(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            device=None,
            dtype=None,
            rope_theta=100000,
            rope_max_seq_len=256,
            attention_backend="vanilla"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.d_head = int(self.d_model/self.n_heads)
        self.attention_backend = attention_backend

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_project = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RoPE(rope_theta, self.d_head, rope_max_seq_len,device=device)
        self.device=device

    def forward(self, x):
        B,T,C = x.shape
        assert C == self.d_model
        q = self.q_proj(x).reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = self.k_proj(x).reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = self.v_proj(x).reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        q = self.rope(q)
        k = self.rope(k)
        if self.attention_backend == "flash_triton":
            output = flash_attention_triton(q,k,v,is_causal=True)
        elif self.attention_backend == "vanilla":
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)) == True
            output = scaled_dot_product_attention(q,k,v,causal_mask.reshape([1,1,T,T]))
        else:
            raise ValueError(f"unknown attention_backend: {self.attention_backend}")
        output = self.out_project(output.transpose(2,1).reshape(B,T,C))
        return output

class  multihead_self_attention(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            device=None,
            dtype=None,
            rope_theta=100000,
            rope_max_seq_len=256,
            attention_backend="vanilla"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.d_head = int(self.d_model/self.n_heads)
        self.attention_backend = attention_backend

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_project = Linear(d_model, d_model, device=device, dtype=dtype)
        self.device=device

    def forward(self, x):
        B,T,C = x.shape
        assert C == self.d_model
        q = self.q_proj(x).reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = self.k_proj(x).reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = self.v_proj(x).reshape(B,T,self.n_heads,self.d_head).transpose(1,2)
        if self.attention_backend == "flash_triton":
            output = flash_attention_triton(q,k,v,is_causal=True)
        elif self.attention_backend == "vanilla":
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)) == True
            output = scaled_dot_product_attention(q,k,v,causal_mask.reshape([1,1,T,T]))
        else:
            raise ValueError(f"unknown attention_backend: {self.attention_backend}")
        output = self.out_project(output.transpose(2,1).reshape(B,T,C))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None, max_seq_len=256, attention_backend="vanilla"):
        super().__init__()
        self.mha = multihead_self_attention_with_rope(
                d_model,
                num_heads,
                device=device,
                dtype=dtype,
                rope_max_seq_len=max_seq_len,
                attention_backend=attention_backend)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm1 = torch.compile(RMSNorm(d_model, device=device, dtype=dtype))
        self.norm2 = torch.compile(RMSNorm(d_model, device=device, dtype=dtype))

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            d_ff,
            vocab_size,
            context_length, # max_context_length
            num_layers,
            device=None,
            dtype=None,
            attention_backend="vanilla"):
        super().__init__()

        self.embedding = Embedding(
                vocab_size,
                d_model,
                device=device,
                dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
                attention_backend=attention_backend)
            for _ in range(num_layers)])
        self.norm = torch.compile(RMSNorm(d_model, device=device, dtype=dtype))
        self.head = Linear(d_model, vocab_size, device=device, dtype=dtype, optional_weight=self.embedding.embeddings)

        self.num_params = 0
        for param in self.parameters():
            self.num_params += functools.reduce(
                    lambda x,y:x*y,
                    param.shape,
                    1)


    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def cross_entropy(x, targets):
    x = x.reshape(-1, x.shape[-1])
    targets = targets.reshape(-1)
    

    m = x.max(dim=-1, keepdims=True).values
    A = x - m
    B = torch.exp(x - m).sum(axis=-1, keepdims=True)

    A = A[ torch.arange(A.shape[-2], device=A.device), targets ].reshape(-1, 1)
    loss = (-A + torch.log(B)).mean()
    # perplexity = torch.exp(loss)
    return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {
                "alpha": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay
                }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                alpha_t = alpha * math.sqrt(1 - math.pow(betas[1], t)) / (1 - math.pow(betas[0], t))
                p.data -= alpha * weight_decay * p.data
                m = state.get("m", 0)
                state["m"] = betas[0] * m + (1 - betas[0]) * grad
                v = state.get("v", 0)
                state["v"] = betas[1] * v + (1 - betas[1]) * (grad ** 2)
                p.data -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] = t + 1
        return loss

def cosine_learning_rate(t, a_max, a_min, t_warm_up, t_final):
    if t < t_warm_up: return a_max * t / t_warm_up
    if t < t_final: 
        cos = math.cos((t-t_warm_up) / (t_final - t_warm_up) * math.pi)
        return a_min + 0.5 * (1 + cos) * (a_max - a_min)
    return a_min

def gradient_clipping(params, M, eps=1e-6):
    l2 = 0
    for param in params:
        if param.grad is not None:
            l2 += (param.grad.data**2).sum().detach()
    l2 = math.sqrt(l2)
    if l2 > M:
        for param in params:
            if param.grad is not None:
                param.grad.data = param.grad.data * M / (l2 + eps)

def save_checkpoint(model, optimizer, iteration, out):
    obj = {}
    obj['model'] = model.state_dict()
    obj['optimizer'] = optimizer.state_dict()
    obj['iteration'] = iteration
    torch.save(obj, out)

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']
