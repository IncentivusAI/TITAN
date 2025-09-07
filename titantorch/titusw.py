# dependencies inspired by transformers/optimization.py  
import math  
import warnings  
from typing import Callable, Iterable, Tuple  

import torch  
from torch import nn  
from torch.optim import Optimizer  

from transformers.utils.versions import require_version  

from .svd_projector import GaLoreProjector  


class TitusW(Optimizer):  
    """  
    Variant of Titus optimizer with decoupled weight decay, introduced in  
    [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).  

    Parameters:  
        params (`Iterable[nn.parameter.Parameter]`): model parameters to update.  
        lr (`float`, optional, default 0.001): step size.  
        betas (`Tuple[float,float]`, optional, default (0.9, 0.999)): Titus coefficients.  
        eps (`float`, optional, default 1e-6): epsilon for stability.  
        weight_decay (`float`, optional, default 0.0): decoupled L2 penalty.  
        correct_bias (`bool`, optional, default True): flag to correct bias terms.  
        no_deprecation_warning (`bool`, optional, default True): disables warning if True.  
    """  

    def __init__(  
        self,  
        params: Iterable[nn.parameter.Parameter],  
        lr: float = 1e-3,  
        betas: Tuple[float, float] = (0.9, 0.999),  
        eps: float = 1e-6,  
        weight_decay: float = 0.0,  
        correct_bias: bool = True,  
        no_deprecation_warning: bool = True,  
    ):  
        if not no_deprecation_warning:  
            warnings.warn(  
                "This TitusW version is deprecated. Use torch.optim.AdamW or set `no_deprecation_warning=True`.",  
                FutureWarning,  
            )  
        require_version("torch>=1.5.0")  
        if lr < 0.0:  
            raise ValueError(f"Invalid learning rate: {lr}")  
        if not 0.0 <= betas[0] < 1.0:  
            raise ValueError(f"Invalid beta1: {betas[0]}")  
        if not 0.0 <= betas[1] < 1.0:  
            raise ValueError(f"Invalid beta2: {betas[1]}")  
        if not 0.0 <= eps:  
            raise ValueError(f"Invalid epsilon: {eps}")  
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}  
        super().__init__(params, defaults)  

    @torch.no_grad()  
    def step(self, closure: Callable = None):  
        """  
        Executes a single optimization step.  
        """  
        loss = None  
        if closure is not None:  
            loss = closure()  

        for group in self.param_groups:  
            for p in group["params"]:  
                if p.grad is None:  
                    continue  
                grad = p.grad  
                if grad.is_sparse:  
                    raise RuntimeError("TitusW does not support sparse gradients")  

                state = self.state[p]  
                if "step" not in state:  
                    state["step"] = 0  

                # GaLore Projection  
                if "rank" in group:  
                    if "projector" not in state:  
                        state["projector"] = GaLoreProjector(  
                            group["rank"],  
                            update_proj_gap=group["update_proj_gap"],  
                            scale=group["scale"],  
                            proj_type=group["proj_type"],  
                        )  
                    grad = state["projector"].project(grad, state["step"])  

                # State initialization  
                if "exp_avg" not in state:  
                    state["exp_avg"] = torch.zeros_like(grad)  
                    state["exp_avg_sq"] = torch.zeros_like(grad)  

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]  
                beta1, beta2 = group["betas"]  

                state["step"] += 1  

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))  
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)  
                denom = exp_avg_sq.sqrt().add_(group["eps"])  

                step_size = group["lr"]  
                if group["correct_bias"]:  
                    bias_correction1 = 1.0 - beta1 ** state["step"]  
                    bias_correction2 = 1.0 - beta2 ** state["step"]  
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1  

                norm_grad = exp_avg / denom  

                # GaLore Projection Back  
                if "rank" in group:  
                    norm_grad = state["projector"].project_back(norm_grad)  

                p.add_(norm_grad, alpha=-step_size)  

                if group["weight_decay"] > 0.0:  
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))  

        return loss  
