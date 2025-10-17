import torch
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
):
    # total_len = dataset.size
    # # NOTE: Have offset 1 to accomodate the target
    # # minus 1 ensures array won't be out of bounds
    # sample_start_idx = np.random.randint(total_len - context_length - 1, size=(batch_size,))
    # inputs = []
    # targets = []
    # for idx in sample_start_idx:
    #     inputs.append(dataset[idx : idx + context_length])
    #     targets.append(dataset[idx + 1 : idx + 1 + context_length])
    # return torch.tensor(np.array(inputs), device=device).long(), torch.tensor(np.array(targets), device=device).long()

    # 不用for循环，改用广播和切片
    total_len = dataset.shape[0]
    # 一次性向量化采样起点（比 for 循环快）
    starts = np.random.randint(0, total_len - context_length - 1, size=batch_size)
    # 利用广播索引一次取出全部切片
    idx = starts[:, None] + np.arange(context_length)[None, :]  # (batch, seq+1)
    batch = dataset[idx]  # 一次性读取所有 (batch, seq+1) 的数据
    inputs = torch.from_numpy(batch[:, :-1]).long()
    targets = torch.from_numpy(batch[:, 1:]).long()
    return inputs.to(device), targets.to(device)
