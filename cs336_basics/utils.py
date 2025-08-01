import typing
import torch
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    torch.save(
        {
            "iter": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        out
    )
    

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    return obj["iter"]