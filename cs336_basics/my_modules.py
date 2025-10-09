import torch
import torch.nn as nn
import math
import einops

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))
        self.reset_parameters()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x @ self.W.T

    def reset_parameters(self):
        std = math.sqrt(self.in_features + self.out_features)
        nn.init.trunc_normal_(self.W, mean = 0, std = std, a = (-3 * std), b = (3 * std))

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.emb, mean = 0, std = 1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        mean = einops.reduce(x ** 2, "b s d -> b s 1", "mean")
        sqrt = torch.sqrt(mean + self.eps)
        result = x / sqrt * self.gain
        # Return the result in the original dtype
        return result.to(in_dtype)

def silu(in_features: torch.Tensor):
    """Apply SiLU activation function to an input tensor

    Args:
        in_features (torch.Tensor): (..., d_model)

    Returns:
        torch.Tensor: (..., d_model)
    """
    return in_features * torch.sigmoid(in_features)


class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = torch.nn.Parameter(torch.empty(d_ff, d_model))

        self.reset_parameters()

    def forward(self, in_features: torch.Tensor):
        """
        Args:
            in_features (torch.Tensor): (..., d_model)
        Returns:
            torch.Tensor: (..., d_model)
        """
       # gate branch
        g = silu(einops.einsum(in_features, self.w1, "... d, f d -> ... f"))  # (..., d_ff)

        # value branch
        x = einops.einsum(in_features, self.w3, "... d, f d -> ... f")  # (..., d_ff)

        # combine + project back
        out = einops.einsum(g * x, self.w2, "... f, d f -> ... d")  # (..., d_model)
        return out

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.w1, mean=0, std=2 / (self.d_ff + self.d_model))
        torch.nn.init.trunc_normal_(self.w2, mean=0, std=2 / (self.d_model + self.d_ff))
        torch.nn.init.trunc_normal_(self.w3, mean=0, std=2 / (self.d_ff + self.d_model))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert (d_k % 2 == 0)
        self.theta = theta
        m = torch.arange(start=0, end=max_seq_len)
        i = torch.arange(start=0, end=d_k // 2)

        i = 1 / (theta ** (2 * i / d_k))
        angles = m[:, None] * i[None, :]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos_submatrix = self.cos[token_positions]
        sin_submatrix = self.sin[token_positions]
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_even * cos_submatrix - x_odd * sin_submatrix
        x_out[..., 1::2] = x_odd * cos_submatrix + x_even * sin_submatrix
        return x_out
    

def softmax(x: torch.Tensor, i: int):
    x = x - torch.max(input=x, dim=i, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / torch.sum(input=exp_x, dim=i, keepdim=True)

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None):
    """
    mask 等于 True 的时候表示不需要掩码, 等于False的时候表示需要掩码
    """

    d_k = Q.size(-1)

    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        addictive_mask = torch.zeros_like(attention_scores)
        addictive_mask = addictive_mask.masked_fill(~mask.to(torch.bool), -torch.inf)
        attention_scores += addictive_mask
    
    attention_scores = softmax(attention_scores, -1)

    return attention_scores @ V

# class CasualMultiHeadSelfAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, max_seq_len:int = None, theta: float = None, device = None, dtype = None):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.d_v = d_model // num_heads

#         self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
#         self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
#         self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
#         self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

#         if max_seq_len is not None and theta is not None:
#         # Rotary positional embedding
#             self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

#     def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
#         B, L, _ = x.shape

#         # Step 1: 线性映射得到 Q, K, V
#         Q = self.W_Q(x)  # (B, L, d_model)
#         K = self.W_K(x)
#         V = self.W_V(x)

#         # Step 2: 拆分多头
#         # reshape 为 (B, num_heads, L, d_k)
#         Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
#         K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
#         V = V.view(B, L, self.num_heads, self.d_v).transpose(1, 2)

#         if token_positions:
#         # Step 3: 应用 RoPE 到 Q, K（对每个head相同）
#             Q = self.rope(Q, token_positions)
#             K = self.rope(K, token_positions)

#         # Step 4: 构造 causal mask
#         # mask shape: (L, L)，True=允许注意, False=禁止
#         mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
#         # broadcast 到 (B, num_heads, L, L)
#         mask = mask.unsqueeze(0).unsqueeze(0)

#         # Step 5: 计算注意力
#         attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)  # (B, num_heads, L, d_v)

#         # Step 6: 合并多头
#         attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)

#         # Step 7: 输出线性层
#         out = self.W_O(attn_out)
#         return out


class CasualMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len:int = None, theta: float = None, device = None, dtype = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.w_k = torch.nn.Parameter(torch.empty(d_model, d_model))
        self.w_v = torch.nn.Parameter(torch.empty(d_model, d_model))
        torch.nn.init.trunc_normal_(self.w_q, mean=0, std=2 / (self.d_model + self.d_model))
        torch.nn.init.trunc_normal_(self.w_k, mean=0, std=2 / (self.d_model + self.d_model))
        torch.nn.init.trunc_normal_(self.w_v, mean=0, std=2 / (self.d_model + self.d_model))
        self.w_o = torch.nn.Parameter(torch.empty(d_model, d_model))
        torch.nn.init.trunc_normal_(self.w_o, mean=0, std=2 / (self.d_model + self.d_model))
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k = self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions:torch.Tensor = None):
        batch_size, seq_len, d_model = x.shape

        Q = x @ self.w_q.T  
        K = x @ self.w_k.T
        V = x @ self.w_v.T

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq_len,seq_len)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        atten_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        atten_out = atten_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return atten_out @ self.w_o.T
    
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pre_ln = RMSNorm(d_model=d_model)
        self.ffn_pre_ln = RMSNorm(d_model=d_model)
        self.attn = CasualMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)

    def forward(self, x:torch.Tensor):
        x = x + self.attn(self.attn_pre_ln(x))
        y = x + self.ffn(self.ffn_pre_ln(x))
        return y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, 
                 rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.Sequential(*[TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, 
                                                       theta=rope_theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor):
        embeddings = self.token_embeddings(x)
        attn_output = self.layers(embeddings)
        output = self.lm_head(self.ln_final(attn_output))
        # NOTE: We don't compute softmax here and will do it
        # in the loss function
        return output