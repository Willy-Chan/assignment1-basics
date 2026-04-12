import torch
import torch.nn as nn
import einops
from einops import reduce, rearrange
import math

def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:

    # subtract the largest logit for stability
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    stability_logits = logits - max_logits
    
    # cross entropy: exponentiate everything, sum those up, then log
    predicted = torch.log(torch.sum(torch.exp(stability_logits), dim=-1))

    # for every batch, grab the correct vocab col
    target_indices = targets.unsqueeze(-1)
    target_logits = torch.gather(stability_logits, dim=-1, index=target_indices).squeeze(-1)    # back to logits shape

    loss = predicted - target_logits

    return torch.mean(loss)

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
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, betas=(0.8, 0.99), eps=1e-8, weight_decay=0.01, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if betas[0] >= 1.0 or betas[0] < 0.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if betas[1] >= 1.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps" : eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # compute adjusted alpha for iteration t
                alphat_denom = 1 - b1 ** t
                alphat_numerator = 1 - b2 ** t
                alpha_t = lr * (alphat_numerator ** 0.5) / alphat_denom
                
                # apply weight decay
                p.data -= lr * decay * p.data

                # update first and second moment estimates
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * (g**2)   
                state["m"] = m
                state["v"] = v

                # apply moment adjusted weight updates
                p.data -= alpha_t * m / (v.sqrt() + eps)

        
        return loss

def learning_rate_schedule(t, amax, amin, Tw, Tc):
    if t < Tw:
        return (t / Tw) * amax
    
    elif Tw <= t <= Tc:
        cos_factor = 0.5 * (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi))
        return amin + cos_factor * (amax - amin)

    else:
        return amin


def gradient_clipping(params, maxl2norm, eps = 1e-6):
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += torch.sum(p.grad ** 2)
    
    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > maxl2norm:
        scale_down_factor = maxl2norm / (total_norm + eps)

        for p in params:
            if p.grad is not None:
                p.grad.data *= scale_down_factor