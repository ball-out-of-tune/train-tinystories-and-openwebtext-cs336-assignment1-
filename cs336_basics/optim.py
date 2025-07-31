from collections.abc import Callable, Iterable
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        # Initialize params and hyper-param defaults
        # Assume all params are in one param_group with the same defaults
        defaults = {
            "lr": lr,
            "decay": weight_decay,
            "b1": betas[0],
            "b2": betas[1],
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        # NOTE: closure is for re-computing the loss before the optimizer step
        # which we don't need
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # NOTE: Hyper-params are associated with each param group
            lr, decay, b1, b2, eps = group["lr"], group["decay"], group["b1"], group["b2"], group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 1)  # Iteration number starts from 1
                m = state.get("m", torch.zeros_like(param.data))
                v = state.get("v", torch.zeros_like(param.data))
                # Get the gradient of the loss at the current time step
                # grad is already detached from the graph so can use it directly
                grad = param.grad
                # Update the first moment estimate
                m = b1 * m + (1 - b1) * grad
                # Update the second moment estimate
                v = b2 * v + (1 - b2) * grad**2
                # Compute adjusted α for iteration t
                alpha = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                # Update the parameters in-place but don't track the gradient
                # equivalent to do param.data.sub_(...)
                with torch.no_grad():
                    param.sub_(alpha * m / (torch.sqrt(v) + eps))
                    # Apply weight decay
                    param.sub_(lr * decay * param.data)
                # NOTE: Update states in the end
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss


def cosine_annealing_lr_scheduler(
    t: int,
    lr_max: float,
    lr_min: float,
    T_w: int,
    T_c: int,
):
    """Cosine annealing

    Args:
        t (int): step
        lr_max (float): max learning rate
        lr_min (float): min learning rate
        T_w (int): warm-up step
        T_c (int): cosine-annealing step

    Returns:
        float: current lr
    """
    if t < T_w:
        # Stage 1: Warm-up stage: gradually increase to max learning rate
        return t / T_w * lr_max
    elif t <= T_c:
        # Stage 2: Gradually reduce learning rate from max to min
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi))
    else:
        # Step 3: Keep min learning rate
        return lr_min


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float):
    all_grads = []
    for p in params:
        if p.grad is None:
            continue
        all_grads.append(p.grad.flatten())

    # NOTE: Compute L2 Norm on flattend gradient from all parameters
    grad_norm = torch.linalg.norm(torch.concat(all_grads), ord=2)
    if grad_norm > max_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad = p.grad * max_norm / (grad_norm + 1e-6)
