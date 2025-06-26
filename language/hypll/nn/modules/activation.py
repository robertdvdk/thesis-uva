from torch.nn import Module
from torch.nn.functional import relu
import torch

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space


class HReLU(Module):
    def __init__(self, manifold: Manifold) -> None:
        super(HReLU, self).__init__()
        self.manifold = manifold

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        return op_in_tangent_space(
            op=relu,
            manifold=self.manifold,
            input=input,
        )


class HypReLU(Module):
    def __init__(self, manifold: Manifold) -> None:
        super(HypReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        space = x[..., 1:].tensor
        space = relu(space)
        time = torch.sqrt(
            torch.norm(space, dim=-1) ** 2 + 1 / self.manifold.c()
        ).unsqueeze(-1)
        output = torch.cat([time, space], dim=-1)
        return ManifoldTensor(data=output, manifold=self.manifold, man_dim=x.man_dim)
