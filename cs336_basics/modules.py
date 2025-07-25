import torch
import logging
from einops import rearrange, repeat

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
        for k in range(self.d_k // 2):
            cos_vals = self.r_cos[token_positions, k]  # (seq_len,)
            sin_vals = self.r_sin[token_positions, k]  # (seq_len,)
            # Apply rotation matrix
            x[..., 2 * k], x[..., 2 * k + 1] = (
                cos_vals * x[..., 2 * k] - sin_vals * x[..., 2 * k + 1],
                sin_vals * x[..., 2 * k] + cos_vals * x[..., 2 * k + 1],
            )
        return x


def softmax(x: torch.Tensor, dim: int):
    """
    Apply softmax on x along a given dimension
    """
    # NOTE: Shift all values to <= 0 to avoid overflow with exp
    x = x - torch.max(x)
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True) # Keep the dim to broadcast divide operation
    
        
