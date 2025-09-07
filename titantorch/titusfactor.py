# pulled dependencies inspired by transformers/optimization.py  
import math  

import torch  
from torch import nn  
from torch.optim import Optimizer  

from transformers.utils.versions import require_version  

from .svd_projector import GaLoreProjector  


class IncentivusAdafactor(Optimizer):  
    """  
    PyTorch version of Adafactor, usable as a substitute for Adam.  
    Reference work: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235  

    Notes:  
    - Adjusts learning rate internally depending on `scale_parameter`, `relative_step`, and `warmup_init`.  
    - To control learning rate externally, set `scale_parameter=False` and `relative_step=False`.  

    Arguments:  
        params (`Iterable[nn.parameter.Parameter]`): parameters or parameter groups.  
        lr (`float`, optional): external learning rate.  
        eps (`Tuple[float, float]`, optional, default `(1e-30, 0.001)`): constants for stability.  
        clip_threshold (`float`, optional, default `1.0`): RMS threshold for gradient updates.  
        decay_rate (`float`, optional, default `-0.8`): factor for running averages.  
        beta1 (`float`, optional): factor for first moment.  
        weight_decay (`float`, optional, default `0.0`): L2 regularization.  
        scale_parameter (`bool`, optional, default `True`): scales LR by RMS.  
        relative_step (`bool`, optional, default `True`): time-based LR if true.  
        warmup_init (`bool`, optional, default `False`): modifies LR behavior when warming up.  

    Recommended for T5 finetuning:  
    - Always apply learning rate warmup.  
    - Use clip_threshold=1.0.  
    - Disable relative updates.  
    - Use scale_parameter=False.  
    """  

    def __init__(  
        self,  
        params,  
        lr=None,  
        eps=(1e-30, 1e-3),  
        clip_threshold=1.0,  
        decay_rate=-0.8,  
        beta1=None,  
        weight_decay=0.0,  
        scale_parameter=True,  
        relative_step=True,  
        warmup_init=False,  
    ):  
        require_version("torch>=1.5.0")  
        if lr is not None and relative_step:  
            raise ValueError("Cannot use manual `lr` with `relative_step=True`")  
        if warmup_init and not relative_step:  
            raise ValueError("`warmup_init=True` requires `relative_step=True`")  

        defaults = {  
            "lr": lr,  
            "eps": eps,  
            "clip_threshold": clip_threshold,  
            "decay_rate": decay_rate,  
            "beta1": beta1,  
            "weight_decay": weight_decay,  
            "scale_parameter": scale_parameter,  
            "relative_step": relative_step,  
            "warmup_init": warmup_init,  
        }  
        super().__init__(params, defaults)  

    @staticmethod  
    def _get_lr(param_group, param_state):  
        lr_val = param_group["lr"]  
        if param_group["relative_step"]:  
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2  
            lr_val = min(min_step, 1.0 / math.sqrt(param_state["step"]))  
        param_scale = 1.0  
        if param_group["scale_parameter"]:  
            param_scale = max(param_group["eps"][1], param_state["RMS"])  
        return param_scale * lr_val  

    @staticmethod  
    def _get_options(param_group, param_shape):  
        factored = len(param_shape) >= 2  
        use_first_moment = param_group["beta1"] is not None  
        return factored, use_first_moment  

    @staticmethod  
    def _rms(tensor):  
        return tensor.norm(2) / (tensor.numel() ** 0.5)  

    @staticmethod  
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):  
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()  
        return torch.mul(r_factor, c_factor)  

    @torch.no_grad()  
    def step(self, closure=None):  
        loss = None  
        if closure is not None:  
            loss = closure()  

        for group in self.param_groups:  
            for p in group["params"]:  
                if p.grad is None:  
                    continue  
                grad = p.grad  
                if grad.dtype in {torch.float16, torch.bfloat16}:  
                    grad = grad.float()  
                if grad.is_sparse:  
                    raise RuntimeError("IncentivusAdafactor does not support sparse gradients.")  

                state = self.state[p]  

                if "step" not in state:  
                    state["step"] = 0  

                # Projection via GaLore  
                if "rank" in group:  
                    if "projector" not in state:  
                        state["projector"] = GaLoreProjector(  
                            group["rank"],  
                            update_proj_gap=group["update_proj_gap"],  
                            scale=group["scale"],  
                            proj_type=group["proj_type"],  
                        )  
                    grad = state["projector"].project(grad, state["step"])  

                grad_shape = grad.shape  
                factored, use_first_moment = self._get_options(group, grad_shape)  

                # Initialize state  
                if "RMS" not in state:  
                    state["step"] = 0  
                    if use_first_moment:  
                        state["exp_avg"] = torch.zeros_like(grad)  
                    if factored:  
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)  
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)  
                    else:  
                        state["exp_avg_sq"] = torch.zeros_like(grad)  
                    state["RMS"] = 0  
                else:  
                    if use_first_moment:  
                        state["exp_avg"] = state["exp_avg"].to(grad)  
                    if factored:  
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)  
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)  
                    else:  
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)  

                p_data_fp32 = p  
                if p.dtype in {torch.float16, torch.bfloat16}:  
                    p_data_fp32 = p_data_fp32.float()  

                state["step"] += 1  
                state["RMS"] = self._rms(p_data_fp32)  
                lr = self._get_lr(group, state)  

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])  
                update = (grad**2) + group["eps"][0]  

                if factored:  
                    exp_avg_sq_row = state["exp_avg_sq_row"]  
                    exp_avg_sq_col = state["exp_avg_sq_col"]  
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))  
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))  
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)  
                    update.mul_(grad)  
                else:  
                    exp_avg_sq = state["exp_avg_sq"]  
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))  
                    update = exp_avg_sq.rsqrt().mul_(grad)  

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))  
                update.mul_(lr)  

                if use_first_moment:  
                    exp_avg = state["exp_avg"]  
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))  
                    update = exp_avg  

                # Projection back  
                if "rank" in group:  
                    update = state["projector"].project_back(update)  

                if group["weight_decay"] != 0:  
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))  

                p_data_fp32.add_(-update)  

                if p.dtype in {torch.float16, torch.bfloat16}:  
                    p.copy_(p_data_fp32)  

        return loss  
