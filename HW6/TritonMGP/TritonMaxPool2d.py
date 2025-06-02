import torch
import torch.nn as nn

import builtins
_original_import = builtins.__import__
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "torch" in name or "nn" in name or "functional" in name:
        raise ImportError(f"Importing '{name}' is not allowed in this file.")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = restricted_import
from .kernel.MaxPool2d import triton_maxpool2d
builtins.__import__ = _original_import


class TritonMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(TritonMaxPool2d, self).__init__()
        self.kernel_size = kernel_size 
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return triton_maxpool2d(x, kernel_size=(self.kernel_size, self.kernel_size), stride=(self.stride, self.stride), padding=(self.padding, self.padding))  