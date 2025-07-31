"""
This module contains the PyTorch model definitions for a Hyperbolic
sequence-to-sequence Transformer, built as an analogue to the Euclidean version.
"""

import torch.nn as nn

from hypll.manifolds.base import Manifold
from hypll.tensors import ManifoldTensor
from hypll.nn import HLinear, HMultiHeadAttention


class HToy(nn.Module):
    def __init__(self, dim: int, manifold: Manifold, impl: str):
        super().__init__()

        self.manifold = manifold
        if impl == "chen":
            self.model = nn.ModuleList([HLinear(in_features=dim, out_features=dim+1, manifold=manifold, bias=True, impl=impl) for _ in range(1)])
        else:
            self.lin = HLinear(in_features=dim, out_features=dim, manifold=manifold, bias=True, impl=impl)

    def forward(self, x: ManifoldTensor):
        x = self.lin(x)
        return x
