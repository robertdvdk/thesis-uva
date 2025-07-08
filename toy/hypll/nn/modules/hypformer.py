import torch.nn as nn
import torch.nn.functional
import torch.nn.init as init
import math

from hypll.manifolds.hyperboloid.math.diffgeom import expmap0
from hypll.manifolds.hyperboloid.math.linalg import lorentz_dot
from hypll.tensors import ManifoldParameter, ManifoldTensor


class HypLayerNorm(nn.Module):
    def __init__(self, manifold, in_features):
        super(HypLayerNorm, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.layer = nn.LayerNorm(self.in_features, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x):
        x = x.tensor

        x_space = x[..., 1:]
        x_space = self.layer(x_space)

        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + 1).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return ManifoldTensor(data=x, manifold=self.manifold, man_dim=-1)


class HypNormalization(nn.Module):
    def __init__(self):
        super(HypNormalization, self).__init__()

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + 1).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        return x


class HypActivation(nn.Module):
    def __init__(self, activation, manifold):
        super(HypActivation, self).__init__()
        self.activation = activation
        self.manifold = manifold

    def forward(self, x):
        x_space = x[..., 1:].tensor
        x_space = self.activation(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + 1).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return ManifoldTensor(data=x, manifold=self.manifold, man_dim=-1)


class HypDropout(nn.Module):
    def __init__(self, dropout):
        super(HypDropout, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: ManifoldTensor):
        if not self.training:
            return x
        x_space = x.tensor[..., 1:]
        x_space_dropped = self.dropout(x_space)
        c = x.manifold.c()
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + 1.0 / c).sqrt()
        new_tensor_data = torch.cat([x_time, x_space_dropped], dim=-1)
        return ManifoldTensor(data=new_tensor_data, manifold=x.manifold, man_dim=x.man_dim)


class HypLinear(nn.Module):
    """
    Parameters:
        manifold (manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """

    def __init__(self, manifold, in_features, out_features, bias=True):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features + 1  # + 1 for time dimension
        self.out_features = out_features
        self.bias = bias

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = x.tensor
        x_space = self.linear(x)

        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + 1).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        return ManifoldTensor(data=x, manifold=self.manifold, man_dim=-1)


class HypCLS(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold = manifold
        cls_emb = torch.randn((self.out_channels, self.in_channels + 1)) * (1. / math.sqrt(self.in_channels + 1))
        cls_emb = expmap0(cls_emb, c=self.manifold.c, man_dim=-1)
        self.cls = ManifoldParameter(cls_emb, self.manifold, man_dim=-1, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x, return_type='neg_dist'):
        dist = -2 * 1 - 2 * lorentz_dot(x.tensor, self.cls.tensor) + self.bias
        if return_type == 'neg_dist':
            return - dist
        elif return_type == 'prob':
            return 10 / (1.0 + dist)
        elif return_type == 'neg_log_prob':
            return - 10 * torch.log(1.0 + dist)
        else:
            raise NotImplementedError