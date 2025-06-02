import torch
import torch.nn as nn

import builtins
_original_import = builtins.__import__
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "torch" in name or "nn" in name or "functional" in name:
        raise ImportError(f"Importing '{name}' is not allowed in this file.")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = restricted_import
from .kernel.ReLU import triton_relu 
builtins.__import__ = _original_import

class TritonReLU(nn.Module):
    def __init__(self, inplace=False):
        super(TritonReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return triton_relu(x, self.inplace)
