import torch
import logging

logger = logging.getLogger(__name__)


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
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
