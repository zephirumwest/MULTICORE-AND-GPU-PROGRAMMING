import torch
import torch.nn as nn

import builtins
_original_import = builtins.__import__
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if "torch" in name or "nn" in name or "functional" in name:
        raise ImportError(f"Importing '{name}' is not allowed in this file.")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = restricted_import
from .kernel.Conv2d import triton_conv2d
builtins.__import__ = _original_import

class TritonConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         1, 1, bias, 'zeros', 'cuda', torch.float32)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        if bias:
            raise NotImplementedError("Bias not supported yet")

    def forward(self, x):
        return triton_conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=(1, 1))