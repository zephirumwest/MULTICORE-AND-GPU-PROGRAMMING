import torch
import torch.nn as nn

import builtins
_original_import = builtins.__import__
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "torch" in name or "nn" in name or "functional" in name:
        raise ImportError(f"Importing '{name}' is not allowed in this file.")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = restricted_import
from .kernel.BatchNorm2d import triton_bn2d
builtins.__import__ = _original_import

class TritonBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_func = None,
        device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, device, dtype)
        self.act_func = act_func

    def forward(self, input):
        self._check_input_dim(input)
        return triton_bn2d(input,
                                       self.weight, self.bias,
                                       self.running_mean, self.running_var,
                                       self.momentum, self.eps)
