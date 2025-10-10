import torch
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
):
    total_len = dataset.size
    # NOTE: Have offset 1 to accomodate the target
    sample_start_idx = np.random.randint(total_len - context_length, size=(batch_size,))
    inputs = []
    targets = []
    for idx in sample_start_idx:
        inputs.append(dataset[idx : idx + context_length])
        targets.append(dataset[idx + 1 : idx + 1 + context_length])
    return torch.tensor(np.array(inputs), device=device).long(), torch.tensor(np.array(targets), device=device).long()
