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
        # token_ids: (batch_size, sequence_length)
        # self.emb: (vocab_size, d_model)
        return torch.stack([self.emb[token_ids[i], :] for i in range(token_ids.size(0))], dim=0)

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.emb, mean=0, std=2 / (self.num_embeddings + self.embedding_dim))
