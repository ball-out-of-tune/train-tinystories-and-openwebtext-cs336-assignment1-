import torch
def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):

    """
    Args:
        logits: (batch_size, vocal_size)
        targets: (batch_size,)

    具体公式，见文档中公式 16 17
    """
    logits = logits - torch.max(input=logits, dim=-1, keepdim=True).values

    # (batch_size, )的分子
    minus_log = -logits[torch.arange(0, targets.size(0)), targets]

    # (batch_size, )的分母
    sum = torch.log(torch.sum(torch.exp(logits), dim=-1))

    return torch.mean(minus_log + sum)