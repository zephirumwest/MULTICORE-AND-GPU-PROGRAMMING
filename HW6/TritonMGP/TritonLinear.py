import torch
import torch.nn as nn

import builtins
_original_import = builtins.__import__
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "torch" in name or "nn" in name or "functional" in name:
        raise ImportError(f"Importing '{name}' is not allowed in this file.")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = restricted_import
from .kernel.Linear import triton_linear
builtins.__import__ = _original_import

class TritonLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func = None,
        device = 'cuda',
        dtype: torch.dtype = torch.float16,
        ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_func = act_func

    def forward(self, x):
        return triton_linear(x.to(torch.float16), self.weight.T.contiguous(), self.bias)