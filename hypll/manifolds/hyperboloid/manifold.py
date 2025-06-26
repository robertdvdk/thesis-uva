import functools
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, empty
from torch.nn import Parameter
from torch.nn.common_types import _size_2_t
from torch.nn.functional import unfold

from hypll.manifolds.base import Manifold
from hypll.manifolds.euclidean import Euclidean
from hypll.manifolds.poincare_ball.curvature import Curvature
from hypll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor
from hypll.utils.math import beta_func
from hypll.utils.tensor_utils import (
    check_dims_with_broadcasting,
    check_if_man_dims_match,
)

from .math.diffgeom import (
    cdist,
    dist,
    expmap,
    expmap0,
    logmap,
    logmap0,
    project,
)
from .math.linalg import (
    lorentz_fully_connected,
    squared_lorentzian_distance,
    lorentz_patch_embedding,
)
from .math.stats import midpoint


class Hyperboloid(Manifold):
    """Class representing the Hyperboloid model of hyperbolic space.

    Implementation based on the geoopt implementation, but changed to use
    hyperbolic torch functions.

    Attributes:
        c:
            Curvature of the manifold.

    """

    def __init__(self, c: Curvature):
        """Initializes an instance of Hyperboloid manifold.

        Examples:
            >>> from hypll.manifolds.poincare_ball import PoincareBall, Curvature
            >>> curvature = Curvature(value=1.0)
            >>> manifold = Manifold(c=curvature)

        """
        super(Hyperboloid, self).__init__()
        self.c = c

    def project(self, x: ManifoldTensor, eps: float = -1.0) -> ManifoldTensor:
        new_tensor = project(x=x.tensor, c=self.c(), dim=x.man_dim)
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def expmap(self, v: TangentTensor) -> ManifoldTensor:
        """Takes in a TangentTensor and returns a ManifoldTensor corresponding to the exponential map
        of the input tensor. If the tangent vectors have corresponding points on the manifold, those
        are used as the "starting point" for the exponential map. Otherwise, the exponential map is
        calculated from the origin.

        Parameters
        ----------
        v : TangentTensor
            TangentTensor to be exponentiated.

        Returns
        -------
        ManifoldTensor
            ManifoldTensor corresponding to the exponential map of the input tensor.
        """
        man_dim = v.broadcasted_man_dim
        if v.manifold_points is None:
            new_tensor = expmap0(v=v.tensor, c=self.c(), man_dim=man_dim)
        else:
            new_tensor = expmap(
                x=v.manifold_points.tensor, v=v.tensor, c=self.c(), man_dim=man_dim
            )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=man_dim)

    def logmap(
        self, y: ManifoldTensor, x: Optional[ManifoldTensor] = None
    ) -> TangentTensor:
        if x is None:
            man_dim = y.man_dim
            new_tensor = logmap0(y=y.tensor, c=self.c())
        else:
            man_dim = check_dims_with_broadcasting(x, y)
            new_tensor = logmap(x=x.tensor, y=y.tensor, c=self.c())
        return TangentTensor(
            data=new_tensor, manifold_points=x, manifold=self, man_dim=man_dim
        )

    def transp(self, y: Tensor, v: Tensor, dim: int = -1) -> Tensor:
        """
        Parallel-transports v from T_x to T_y on the hyperboloid model.

        Args:
            x: Point on hyperboloid (batchable, Minkowski norm = -1/c).
            y: Another point on hyperboloid (same shape as x).
            v: Tangent vector at x (i.e. <x,v>_M=0).
            c: Curvature.
            dim: Dimension for Minkowski coordinates.

        Returns:
            A new tensor of the same shape as v, tangent at y.
        """
        x = v.manifold_points.tensor
        # <x,y>_M
        xy = self.minkowski_dot(x, y.tensor, dim=dim, keepdim=True)
        # <y,v>_M
        yv = self.minkowski_dot(y.tensor, v.tensor, dim=dim, keepdim=True)

        # denominator = <x,y>_M - 1/c
        denom = xy - 1.0 / self.c()
        tangent_vectors = v.tensor - (yv / denom) * (x + y.tensor)
        # The standard formula for parallel transport on the hyperboloid
        return TangentTensor(
            data=tangent_vectors,
            manifold_points=y,
            manifold=self,
            man_dim=dim,
        )

    def dist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        dim = check_dims_with_broadcasting(x, y)
        return dist(x=x.tensor, y=y.tensor, c=self.c(), dim=dim)

    def fully_connected(
        self,
        x: ManifoldTensor,
        z: ManifoldTensor,
        bias: Optional[Tensor],
        num_heads: Optional[int],
    ) -> ManifoldTensor:
        """
        Computes Lorentzian fully connected layer according to the formulation in Chen et al. (2021).

        Parameters
        ----------
        x : ManifoldTensor
            Input tensor, shape (B, L, D + 1).
        z : ManifoldTensor
            Weight tensor, shape (D, D + 1).
        bias : Optional[Tensor]
            Bias tensor, shape (D).
        num_heads : Optional[int]
            Used when the fully connected layer is used in the context of multi-head attention.

        Returns
        -------
        ManifoldTensor
            Output tensor, shape (B, L, D + 1).
        """
        new_tensor = lorentz_fully_connected(
            x=x.tensor,
            W=z.tensor,
            bias=bias,
            c=self.c(),
            num_heads=num_heads,
        )
        man_dim = (
            x.man_dim + 1 if num_heads > 1 else x.man_dim
        )  # Update manifold dimension in case of multi-head attention.
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=man_dim)

    def multiheadattention(
        self,
        query: ManifoldTensor,
        key: ManifoldTensor,
        value: ManifoldTensor,
        W_Q: torch.nn.Module,
        W_K: torch.nn.Module,
        W_V: torch.nn.Module,
        W_O: torch.nn.Module,
    ) -> ManifoldTensor:
        """
        Computes Lorentzian multi-head attention according to the formulation in Chen et al. (2021).

        Parameters
        ----------
        query : ManifoldTensor
            ManifoldTensor of shape (B, L, D + 1) representing the query tensor.
        key : ManifoldTensor
            ManifoldTensor of shape (B, L, D + 1) representing the key tensor.
        value : ManifoldTensor
            ManifoldTensor of shape (B, L, D + 1) representing the value tensor.
        W_Q : torch.nn.Module
            Hyperboloid linear layer for the query tensor.
        W_K : torch.nn.Module
            Hyperboloid linear layer for the key tensor.
        W_V : torch.nn.Module
            Hyperboloid linear layer for the value tensor.
        W_O : torch.nn.Module
            Hyperboloid linear layer for the output tensor.

        Returns
        -------
        ManifoldTensor
            ManifoldTensor of shape (B, L, D + 1) representing the output tensor.
        """
        num_heads = W_Q.num_heads
        query = W_Q.forward(query)  # (B, L, H, D + 1)
        key = W_K.forward(key)  # (B, L, H, D + 1)
        value = W_V.forward(value)  # (B, L, H, D + 1)
        if num_heads > 1:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        n = torch.tensor(
            query.shape[-1] - 1
        )  # Space dimension of the hyperboloid (time is not counted).
        w = torch.nn.functional.softmax(
            -squared_lorentzian_distance(query.tensor, key.tensor, self.c()) / n.sqrt(),
            dim=-1,
        )  # (B, H, L, L)
        mu = self.midpoint(x=value, w=w, num_heads=num_heads)  # (B, L, D + 1)
        Ox = W_O.forward(mu)  # (B, L, D + 1)
        return Ox

    def patch_embedding(
        self, x: Tensor, z: ManifoldTensor, positional_encoding: Tensor, patch_size: int
    ) -> ManifoldTensor:
        new_tensor = lorentz_patch_embedding(
            x=x,
            weights=z.tensor,
            positional_encoding=positional_encoding,
            c=self.c(),
            patch_size=patch_size,
        )
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=-1)

    def midpoint(
        self,
        x: ManifoldTensor,
        w: Optional[Tensor] = None,
        num_heads: Optional[int] = 1,
    ) -> ManifoldTensor:
        """
        Calculates the (weighted) centroid of a set of points on the hyperboloid, formulated by Law et al. (2019).

        Parameters
        ----------
        x : ManifoldTensor
            Set of points on the hyperboloid, shape (B, L, D + 1).
        w : Optional[Tensor], optional
            Weights to use in midpoint calculation, by default None
        num_heads : Optional[int], optional
            Number of heads in case the midpoint is used in multi-head attention, by default 1

        Returns
        -------
        ManifoldTensor
            _description_
        """
        # print("Mu before", x.tensor[:, :, 0, 1])
        mu = midpoint(
            x=x.tensor,
            c=self.c(),
            w=w,
        )  # (B, H, L, (D / H) + 1)
        if num_heads > 1:
            mu_space = mu[..., 1:].reshape(mu.shape[0], mu.shape[-2], -1)  # (B, L, D)
            mu_time = torch.sqrt(
                torch.norm(mu_space, dim=-1) ** 2 + 1 / self.c()
            ).unsqueeze(-1)  # (B, L, 1)
            mu = torch.cat([mu_time, mu_space], dim=-1)  # (B, L, D + 1)
        man_dim = x.man_dim - 1 if num_heads > 1 else x.man_dim
        return ManifoldTensor(data=mu, manifold=self, man_dim=man_dim)

    def inner(self, u, v, dim=-1, keepdim=False, safe_mode=False):
        return self.minkowski_dot(u.tensor, v.tensor, dim=dim, keepdim=keepdim)

    def minkowski_dot(
        self, x: Tensor, y: Tensor, dim: int = -1, keepdim: bool = False
    ) -> Tensor:
        """
        Computes the Minkowski dot product in (-, +, +, ..., +) signature
        along the specified dimension 'dim'.
        """
        # Split off the first coordinate (time-like) and the remaining space-like coordinates
        time_x, space_x = x.split([1, x.size(dim) - 1], dim=dim)
        time_y, space_y = y.split([1, y.size(dim) - 1], dim=dim)

        # Minkowski dot: - x0*y0 + sum_{i=1 to d} x_i*y_i
        dot = -time_x * time_y + (space_x * space_y).sum(dim=dim, keepdim=keepdim)
        return dot

    def euc_to_tangent(
        self, x: ManifoldTensor, u: ManifoldTensor, dim: int = -1
    ) -> TangentTensor:
        """
        Formula from "Gradient descent in hyperbolic space", by Wilson and Leimester, 2018.
        """
        x_dot_u = (x.tensor * u.tensor).sum(dim=dim, keepdim=True)
        u.tensor.narrow(-1, 0, 1).mul_(-1)
        return TangentTensor(
            data=u.tensor.addcmul(x_dot_u, x.tensor),
            manifold_points=x,
            manifold=self,
            man_dim=dim,
        )

    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> tuple[ManifoldParameter, Optional[Parameter]]:
        weight = ManifoldParameter(
            data=empty(out_features, in_features + 1),
            manifold=Euclidean(),
            man_dim=0,
        )

        if bias:
            b = Parameter(data=empty(out_features))
        else:
            b = None

        return weight, b

    def reset_parameters(
        self, weight: ManifoldParameter, bias: Optional[Parameter]
    ) -> None:
        torch.nn.init.xavier_uniform_(weight.tensor.data)
        weight.tensor.data = weight.tensor.data
        if bias is not None:
            bias.data.zero_()

    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> ManifoldTensor:
        # TODO: may have to cache some of this stuff for efficiency.
        in_channels = input.size(1)
        if len(kernel_size) == 2:
            kernel_vol = kernel_size[0] * kernel_size[1]
        else:
            kernel_vol = kernel_size**2
            kernel_size = (kernel_size, kernel_size)

        beta_ni = beta_func(in_channels / 2, 1 / 2)
        beta_n = beta_func(in_channels * kernel_vol / 2, 1 / 2)

        input = self.logmap(x=None, y=input)
        input.tensor = input.tensor * beta_n / beta_ni
        new_tensor = unfold(
            input=input.tensor,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        new_tensor = TangentTensor(
            data=new_tensor, manifold_points=None, manifold=self, man_dim=1
        )
        return self.expmap(new_tensor)

    def flatten(
        self, x: ManifoldTensor, start_dim: int = 1, end_dim: int = -1
    ) -> ManifoldTensor:
        """Flattens a manifold tensor by reshaping it. If start_dim or end_dim are passed,
        only dimensions starting with start_dim and ending with end_dim are flattend.

        If the manifold dimension of the input tensor is among the dimensions which
        are flattened, applies beta-concatenation to the points on the manifold.
        Otherwise simply flattens the tensor using torch.flatten.

        Updates the manifold dimension if necessary.

        """
        start_dim = x.dim() + start_dim if start_dim < 0 else start_dim
        end_dim = x.dim() + end_dim if end_dim < 0 else end_dim

        # Get the range of dimensions to flatten.
        dimensions_to_flatten = x.shape[start_dim + 1 : end_dim + 1]

        if start_dim <= x.man_dim and end_dim >= x.man_dim:
            # Use beta concatenation to flatten the manifold dimension of the tensor.
            #
            # Start by applying logmap at the origin and computing the betas.
            tangents = self.logmap(None, x)
            n_i = x.shape[x.man_dim]
            n = n_i * functools.reduce(lambda a, b: a * b, dimensions_to_flatten)
            beta_n = beta_func(n / 2, 0.5)
            beta_n_i = beta_func(n_i / 2, 0.5)
            # Flatten the tensor and rescale.
            tangents.tensor = torch.flatten(
                input=tangents.tensor,
                start_dim=start_dim,
                end_dim=end_dim,
            )
            tangents.tensor = tangents.tensor * beta_n / beta_n_i
            # Set the new manifold dimension
            tangents.man_dim = start_dim
            # Apply exponential map at the origin.
            return self.expmap(tangents)
        else:
            flattened = torch.flatten(
                input=x.tensor,
                start_dim=start_dim,
                end_dim=end_dim,
            )
            man_dim = (
                x.man_dim
                if end_dim > x.man_dim
                else x.man_dim - len(dimensions_to_flatten)
            )
            return ManifoldTensor(data=flattened, manifold=x.manifold, man_dim=man_dim)

    def cdist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        return cdist(x=x.tensor, y=y.tensor, c=self.c())

    def cat(
        self,
        manifold_tensors: Union[Tuple[ManifoldTensor, ...], List[ManifoldTensor]],
        dim: int = 0,
    ) -> ManifoldTensor:
        check_if_man_dims_match(manifold_tensors)
        if dim == manifold_tensors[0].man_dim:
            tangent_tensors = [self.logmap(None, t) for t in manifold_tensors]
            ns = torch.tensor([t.shape[t.man_dim] for t in manifold_tensors])
            n = ns.sum()
            beta_ns = beta_func(ns / 2, 0.5)
            beta_n = beta_func(n / 2, 0.5)
            cat = torch.cat(
                [
                    (t.tensor * beta_n) / beta_n_i
                    for (t, beta_n_i) in zip(tangent_tensors, beta_ns)
                ],
                dim=dim,
            )
            new_tensor = TangentTensor(data=cat, manifold=self, man_dim=dim)
            return self.expmap(new_tensor)
        else:
            cat = torch.cat([t.tensor for t in manifold_tensors], dim=dim)
            man_dim = manifold_tensors[0].man_dim
            return ManifoldTensor(data=cat, manifold=self, man_dim=man_dim)
