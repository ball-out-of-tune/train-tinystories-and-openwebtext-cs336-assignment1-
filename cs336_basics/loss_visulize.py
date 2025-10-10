import torch

def cross_entropy_loss_detailed(logits: torch.Tensor, targets: torch.Tensor):
    """
    详细展示交叉熵损失的每一步计算过程
    """
    print("=" * 80)
    print("原始输入")
    print("=" * 80)
    print(f"logits shape: {logits.shape}")
    print(f"logits:\n{logits}\n")
    print(f"targets shape: {targets.shape}")
    print(f"targets:\n{targets}\n")
    
    # 步骤 1: 数值稳定性处理 - 减去每行的最大值
    print("=" * 80)
    print("步骤 1: 数值稳定性处理 (减去每行最大值)")
    print("=" * 80)
    max_logits = torch.max(logits, dim=-1, keepdim=True)
    print(f"每行最大值 shape: {max_logits.values.shape}")
    print(f"每行最大值:\n{max_logits.values}\n")
    
    logits_normalized = logits - max_logits.values
    print(f"归一化后的 logits:\n{logits_normalized}\n")
    
    # 步骤 2: Reshape 为 2D
    print("=" * 80)
    print("步骤 2: Reshape 操作")
    print("=" * 80)
    B, T, C = logits.shape
    print(f"原始形状: B={B}, T={T}, C={C}")
    
    logits_2d = logits_normalized.reshape(B*T, C)
    targets_1d = targets.reshape(B*T)
    print(f"logits_2d shape: {logits_2d.shape}")
    print(f"logits_2d:\n{logits_2d}\n")
    print(f"targets_1d shape: {targets_1d.shape}")
    print(f"targets_1d: {targets_1d}\n")
    
    # 步骤 3: 计算每个样本对应目标类别的 logit 值 (-o_i[x_i+1])
    print("=" * 80)
    print("步骤 3: 提取目标类别的 logit 值")
    print("=" * 80)
    target_logits = logits_2d[torch.arange(targets_1d.size(0)), targets_1d]
    print(f"索引说明:")
    for i in range(B*T):
        print(f"  样本 {i}: logits_2d[{i}, {targets_1d[i].item()}] = {target_logits[i].item():.4f}")
    print(f"\ntarget_logits: {target_logits}\n")
    
    negative_target_logits = -target_logits
    print(f"负的目标 logits (-o_i[x_i+1]): {negative_target_logits}\n")
    
    # 步骤 4: 计算 log-sum-exp
    print("=" * 80)
    print("步骤 4: 计算 log(sum(exp(logits)))")
    print("=" * 80)
    exp_logits = torch.exp(logits_2d)
    print(f"exp(logits_2d):\n{exp_logits}\n")
    
    sum_exp = torch.sum(exp_logits, dim=-1)
    print(f"sum(exp(logits)) 对每行求和: {sum_exp}\n")
    
    log_sum_exp = torch.log(sum_exp)
    print(f"log(sum(exp(logits))): {log_sum_exp}\n")
    
    # 步骤 5: 计算每个样本的损失
    print("=" * 80)
    print("步骤 5: 计算每个样本的损失")
    print("=" * 80)
    individual_losses = negative_target_logits + log_sum_exp
    print(f"每个样本的损失 = -o_i[x_i+1] + log(sum(exp(o_i))):")
    for i in range(B*T):
        print(f"  样本 {i}: {negative_target_logits[i].item():.4f} + {log_sum_exp[i].item():.4f} = {individual_losses[i].item():.4f}")
    print(f"\nindividual_losses: {individual_losses}\n")
    
    # 步骤 6: 计算平均损失
    print("=" * 80)
    print("步骤 6: 计算平均损失")
    print("=" * 80)
    mean_loss = torch.mean(individual_losses)
    print(f"平均损失 = sum(losses) / {B*T}")
    print(f"平均损失 = {individual_losses.sum().item():.4f} / {B*T} = {mean_loss.item():.4f}\n")
    
    # 验证：使用 PyTorch 内置函数
    print("=" * 80)
    print("验证：使用 PyTorch 内置 CrossEntropyLoss")
    print("=" * 80)
    pytorch_loss = torch.nn.functional.cross_entropy(
        logits.reshape(B*T, C), 
        targets_1d
    )
    print(f"PyTorch CrossEntropyLoss: {pytorch_loss.item():.4f}")
    print(f"我们的实现: {mean_loss.item():.4f}")
    print(f"差异: {abs(pytorch_loss.item() - mean_loss.item()):.6f}\n")
    
    return mean_loss


# 测试数据
logits = torch.tensor([
    [[2.0, 1.0, 0.1, 0.5], [1.0, 2.0, 0.5, 0.2], [0.1, 0.2, 3.0, 0.3]],
    [[0.5, 0.1, 2.0, 1.0], [1.5, 0.5, 0.2, 2.0], [0.3, 2.5, 0.5, 0.1]]
])

targets = torch.tensor([
    [0, 1, 2],
    [2, 3, 1]
])

# 执行详细计算
loss = cross_entropy_loss_detailed(logits, targets)

print("=" * 80)
print("计算公式总结")
print("=" * 80)
print("交叉熵损失 = mean( -o_i[x_i+1] + log(sum(exp(o_i))) )")
print("其中:")
print("  - o_i[x_i+1] 是目标类别对应的 logit 值")
print("  - log(sum(exp(o_i))) 是 log-sum-exp 项 (归一化常数)")
print("  - 减去最大值是为了数值稳定性，不影响最终结果")