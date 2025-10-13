import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    Cross Entropy Loss (a.k.a. negative log-likelihood)
    = -log( p(x_1, x_2, ... , x_i) )
    = sum_i [-log( p(x_i+1 | x_1:i) )]
    = sum_i [-log( exp(o_i[x_i+1]) / \sum_j exp(o_i[j]) )]
    = sum_i [-log( exp(o_i[x_i+1]) ) + log( \sum_j exp(o_i[j]) )]
    = sum_i [-o_i[x_i+1] + log( sum_j [exp(o_i[j])] )]
    where i is the token index, j is the feature index
    o_i is the logits, x_i+1 is the targets

    NOTE: We can treah every dimension as batch dimension (including seq dim)
    since we only care about the feature dim for loss computation

    Args:
        logits: (batch_size, seq_len, vocal_size)
        targets: (batch_size, seq_len)
    """
    # NOTE: Subtract the largest value per sample, keep dim to broadcast the max value
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = targets.reshape(B*T)
    return torch.mean(
        # NOTE: specify row and column indices respectively to get the probability of each target
        -logits[torch.arange(targets.size(0)), targets]  # -o_i[x_i+1] => (batch_size,)
        + torch.log(torch.sum(torch.exp(logits), dim=-1))  # log( \sum_j exp(o_i[j]) ) => (batch_size,)
    )

# # ===== 测试数据 =====
# batch_size = 2
# seq_len = 3
# vocab_size = 5

# logits = torch.randn(batch_size, seq_len, vocab_size)  # 随机 logits
# targets = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机 targets

# loss = cross_entropy_loss(logits, targets)
# print("loss:", loss)
# print("logits.shape:", logits.shape)
# print("targets.shape:", targets.shape)