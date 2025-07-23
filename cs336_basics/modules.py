import torch
import logging

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

        # NOTE: Initialize row-major weight matrix, meaning each row's (i.e. in_features dim) memory will be contiguous
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
        # self.emb: (vocab_size, d_model)
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
        
        # Learnable scaling factor for all tokens
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
        rms = torch.sqrt((1 / self.d_model) * torch.sum(x ** 2, dim=-1) + self.eps).unsqueeze(-1)  # (batch_size, sequence_length, 1) 
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
