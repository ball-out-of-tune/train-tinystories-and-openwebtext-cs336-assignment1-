import torch
def softmax(x: torch.Tensor, i: int):
    # 数值稳定性：在指定维度上减去最大值
    max_vals = torch.max(x, dim=i, keepdim=True).values
    shifted = x - max_vals
    
    # 计算指数
    exp_vals = torch.exp(shifted)
    
    # 在指定维度上求和并归一化
    sum_exp = torch.sum(exp_vals, dim=i, keepdim=True)
    
    return exp_vals / sum_exp

print(softmax(torch.Tensor([[1, 3, 5], [6, 4, 2], [10, 11, 12]]), 1))