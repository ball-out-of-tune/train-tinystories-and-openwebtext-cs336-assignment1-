import torch
import logging
import einops

logger = logging.getLogger(__name__)


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features

        # NOTE: Weight matrix is initialized in row-major, meaning each row's (i.e. in_features dim) memory will be contiguous
        # This makes matmul more efficient
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.W, mean=0, std=2 / (self.out_features + self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (..., dim_in)
        Returns:
            torch.Tensor: (..., dim_out)
        """
        return x @ self.W.T


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # vocab_size
        embedding_dim: int,  # d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, sequence_length)
        Returns:
            torch.Tensor: (batch_size, sequence_length, d_model)
        """
        # Recall that self.emb: (vocab_size, d_model)
        return torch.stack([self.emb[token_ids[i], :] for i in range(token_ids.size(0))], dim=0)

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.emb, mean=0, std=2 / (self.num_embeddings + self.embedding_dim))


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # NOTE: This is an affine transformation of the normalized features
        # applied on all tokens
        self.gain = torch.nn.Parameter(torch.ones(self.d_model))  # (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, sequence_length, d_model)
        Returns:
            torch.Tensor: (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # NOTE: Root mean square is computed along the feature dimension, meaning each token
        # is normalized w.r.t. its own features. The gain is applied on each feature for all tokens.
        rms = torch.sqrt((1 / self.d_model) * torch.sum(x**2, dim=-1) + self.eps).unsqueeze(
            -1
        )  # (batch_size, sequence_length, 1)
        rms_norm = x / rms * self.gain
        return rms_norm.to(in_dtype)


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
        g = silu(in_features @ self.w1.T)  # gate: (..., d_ff)
        x = in_features @ self.w3.T  # activation: (..., d_ff)
        return (g * x) @ self.w2.T  # (..., d_model)

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.w1, mean=0, std=2 / (self.d_ff + self.d_model))
        torch.nn.init.trunc_normal_(self.w2, mean=0, std=2 / (self.d_model + self.d_ff))
        torch.nn.init.trunc_normal_(self.w3, mean=0, std=2 / (self.d_ff + self.d_model))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        # Pre-compute rotation matrix entries
        assert d_k % 2 == 0, "d_k must be even"
        # The rotaion angle depends on token index i and feature index k
        # So we pre-populate some matrices for element-wise multiplication
        # NOTE: In the assignment, k starts from 1 to d/2. We change it to [0, d/2-1]
        # and compute the cos/ sin accordingly. Turns out this is correct...
        i = torch.arange(max_seq_len).unsqueeze(-1).expand(-1, d_k // 2)  # (max_seq_len, d_k // 2)
        k = torch.arange(d_k // 2).unsqueeze(0).expand(max_seq_len, -1)  # (max_seq_len, d_k // 2)

        r_cos = torch.cos(i / theta ** (2 * k / d_k))  # (max_seq_len, d_k // 2)
        r_sin = torch.sin(i / theta ** (2 * k / d_k))  # (max_seq_len, d_k // 2)

        self.register_buffer("r_cos", r_cos, persistent=False)
        self.register_buffer("r_sin", r_sin, persistent=False)

        self.d_k = d_k
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation matrix operation on each pair of entries from input's d_k dim.
        Rotation matrix is defined as
            [cos(theta_i_k), -sin(theta_i_k)]
            [sin(theta_i_k),  cos(theta_i_k)]
        Args:
            x (torch.Tensor): (..., seq_len, d_k)
            token_positions (torch.Tensor): (..., seq_len)

        Returns:
            torch.Tensor: (..., seq_len, d_k)
        """
        # NOTE: Iterate each pair of embedding elements and compute rotation
        # using the cos/ sin buffer, so that we don't need to construct the full rotation matrix
        rotated = torch.empty(x.shape)
        for k in range(self.d_k // 2):
            cos_vals = self.r_cos[token_positions, k]  # (seq_len,)
            sin_vals = self.r_sin[token_positions, k]  # (seq_len,)
            # Apply rotation matrix
            rotated[..., 2 * k], rotated[..., 2 * k + 1] = (
                cos_vals * x[..., 2 * k] - sin_vals * x[..., 2 * k + 1],
                sin_vals * x[..., 2 * k] + cos_vals * x[..., 2 * k + 1],
            )
        return rotated


def softmax(x: torch.Tensor, dim: int):
    """
    Apply softmax on x along a given dimension
    """
    # NOTE: Shift all values to <= 0 to avoid overflow with exp
    x = x - torch.max(x)
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)  # Keep the dim to broadcast divide operation


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Args:
        q (torch.Tensor): (batch_size, ..., seq_len_q, d_k)
        k (torch.Tensor): (batch_size, ..., seq_len_k, d_k)
        v (torch.Tensor): (batch_size, ..., seq_len_k, d_v)
        mask (torch.Tensor): (seq_len_q, seq_len_k)
            **Note that mask is True if attention is allowed, False otherwise**
    """
    assert q.size(-1) == k.size(-1)  # Feature dim must match between q and k
    assert k.size(-2) == v.size(-2)  # Sequence length must match between k and v

    prod = (
        # NOTE: Can also use einops.einsum(q, k, "... q d,... k d->... q k"), but slower for some reason
        torch.einsum("...qd,...kd->...qk", q, k) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.int))
    )
    # NOTE: Make masked entry's probability to be 0 after softmax, thus no attention is paid
    prod[..., ~mask.to(torch.bool)] = -torch.inf
    sm = softmax(prod, dim=-1)  # (seq_len_q, seq_len_k) where each row along seq_len_k is normalized
    return sm @ v  # Linear combination of vs for each q, using k as "probability": (seq_len_q, d_v)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_enabled = max_seq_len is not None and theta is not None

        # NOTE: Following Vaswani et al. [2017], we set d_k = d_v = d_model / num_heads
        # but it does not have to be. For example, we can project from d_model to h * d_k > d_model
        self.d_q = d_model // num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # NOTE: Similar to Linear, we build a row-major weight matrix on input dimension for efficiency
        self.w_q = torch.nn.Parameter(torch.empty(num_heads * self.d_q, d_model))
        self.w_k = torch.nn.Parameter(torch.empty(num_heads * self.d_k, d_model))
        self.w_v = torch.nn.Parameter(torch.empty(num_heads * self.d_v, d_model))
        self.w_o = torch.nn.Parameter(torch.empty(d_model, num_heads * self.d_v))

        self.reset_parameters()

        if self.rope_enabled:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        Args:
            x (torch.Tensor): (..., seq_len, d_model)
            token_positions (torch.Tensor): (..., seq_len)
        """
        assert x.size(-1) == self.d_model
        seq_len = x.size(-2)

        # Step 1: Apply single matrix multiplication to project q, k, v vectors for all heads
        # Each output feature vector is a linear combination of weight values and input x
        w_qkv = torch.concat(
            (self.w_q.T, self.w_k.T, self.w_v.T), dim=-1
        )  # (num_heads * d_q + num_heads * d_k + num_heads * d_v, d_model)
        qkv = x @ w_qkv  # (..., seq_len, num_heads * d_q + num_heads * d_k + num_heads * d_v)

        # Step 2: Slice q, k, v for each head.
        # NOTE: Need to split q, k, v first due to concat order
        # - Split q, k, v
        q, k, v = qkv.chunk(3, dim=-1)  # # (..., num_heads, seq_len, d_q)
        # # - Split head
        q = einops.rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)  # (..., num_heads, seq_len_q, d_q)
        k = einops.rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)  # (..., num_heads, seq_len_k, d_k)
        v = einops.rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)  # (..., num_heads, seq_len_v, d_v

        if self.rope_enabled:
            # Step 3: Apply RoPE to the query and key vectors, but not the value vectors.
            # NOTE: The same RoPE rotation should be applied to the query and key vectors for each head
            # Thus each RoPE embedding should have d_k length
            if token_positions is None:
                token_positions = torch.arange(seq_len)
                logger.info(f"Token position is not provided. Assume {token_positions}")
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)

        # Step 4: Create causal attention mask (same mask)
        # NOTE: Set the diagnal and every entry below it as 1 for causal attention
        # An example of 5x5 is as following
        # 1 0 0 0 0
        # 1 1 0 0 0
        # 1 1 1 0 0
        # 1 1 1 1 0
        # 1 1 1 1 1
        # Conveniently, we can directly transpose an upper-triangular matrix like below
        # since sequence len is the same between Q and K for self-attention
        mask = torch.triu(torch.ones(seq_len, seq_len)).T.to(torch.bool)

        # Step 5: Run scale dot product attention on each head
        out = scaled_dot_product_attention(q, k, v, mask)  # (..., num_heads, seq_len, d_v)

        # Step 6: Combine output from all heads and project to output dimension
        out = einops.rearrange(out, "... h s d -> ... s (h d)")  # (..., seq_len, num_heads * d_v)
        return out @ self.w_o.T  # (..., seq_len, d_model)

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.w_q, mean=0, std=2 / (self.d_model + self.d_model))
        torch.nn.init.trunc_normal_(self.w_k, mean=0, std=2 / (self.d_model + self.d_model))
        torch.nn.init.trunc_normal_(self.w_v, mean=0, std=2 / (self.d_model + self.d_model))


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()
        self.attn_pre_ln = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ffn_pre_ln = RMSNorm(d_model=d_model)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor):
        attn_output = x + self.attn(self.attn_pre_ln(x))
        return attn_output + self.ffn(self.ffn_pre_ln(attn_output))


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = torch.nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor):
        embeddings = self.token_embeddings(x)
        attn_output = self.layers(embeddings)
        output = self.lm_head(self.ln_final(attn_output))
        # NOTE: We don't compute softmax here and will do it
        # in the loss function
        return output
