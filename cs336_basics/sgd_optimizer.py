from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss
    

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=10)
# print("参数组数量:", len(opt.param_groups))
# print("第一个参数组:", opt.param_groups[0])
# print("参数组中的参数数量:", len(opt.param_groups[0]['params']))
# print("学习率:", opt.param_groups[0]['lr'])  # 输出: 1000

# print("初始 weights.grad:", weights.grad)  # None

# for t in range(3):  # 只运行3次来观察
#     opt.zero_grad()
#     print(f"\n--- 迭代 {t} ---")
    
#     # 计算损失
#     loss = (weights**2).mean()
#     print(f"计算损失: {loss.item()}")
#     print(f"损失计算后 weights.grad: {weights.grad}")  # 可能还是None
    
#     # 反向传播
#     loss.backward()
#     print(f"反向传播后 weights.grad 形状: {weights.grad.shape}")
#     print(f"weights.grad 的部分值: {weights.grad[0, 0:3]}")
    
#     # 优化器更新
#     opt.step()
#     print(f"优化器更新后 loss: {loss.item()}")  # loss值不变，但weights已更新
#     print(f"更新后 weights 的部分值: {weights.data[0, 0:3]}")

for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.


"""
asnwer is:
in 100 timesteps, 1 and 10 won't converge to 0, 10 converges faster than 1; 100 will converge to 0;
1000 will diverge to inf 
"""

"""
# 创建模型和优化器
model = torch.nn.Linear(10, 1)
optimizer = SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    # 还可以加上数据采样, 选取全部数据或者选取部分数据
    # 前向传播
    output = model(input_data)
    loss = criterion(output, target)
    
    # 反向传播
    loss.backward()
    
    # 参数更新
    optimizer.step()
    
    # 梯度清零
    optimizer.zero_grad()
"""