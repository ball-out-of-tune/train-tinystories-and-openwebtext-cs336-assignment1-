import torch
from collections.abc import Callable, Iterable
import math
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "epsilon": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params=params, defaults=defaults)

    def step(self, closure: Callable | None = None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, beta1, beta2, epsilon, weight_decay = group["lr"], group["beta1"], group["beta2"], group["epsilon"], group["weight_decay"]
            for param in group["params"]:
                if param.grad == None:
                    continue
                state = self.state[param]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(param.data))
                v = state.get("v", torch.zeros_like(param.data))

                grad = param.grad
                m = beta1 * m  + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                with torch.no_grad():
                    param.sub_(lr_t * m / (torch.sqrt(v) + epsilon))
                    param.sub_(lr * weight_decay * param.data)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


