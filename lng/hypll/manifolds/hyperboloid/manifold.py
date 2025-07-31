import functools
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, empty
from torch.nn import Parameter
import torch.nn as nn
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
    euc_to_tangent
)
from .math.linalg import (
    squared_lorentzian_distance,
)
from .math.stats import midpoint

import torch.nn.functional as F
import sys

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
        new_tensor = project(x=x.tensor, c=self.c())
        return ManifoldTensor(data=new_tensor, manifold=self, man_dim=x.man_dim)

    def expmap(self, v: TangentTensor, prnt=False) -> ManifoldTensor:
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
                x=v.manifold_points.tensor, v=v.tensor, c=self.c(), prnt=prnt
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

    def fully_connected(self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor], l: float, dropout: nn.Module, impl: str):
        if impl == "tangent":
            # space = x @ z.tensor.T
            space = self.logmap(y=x).tensor @ z.tensor.T
            if bias is not None:
                space += bias
            time = torch.zeros(space.shape[:-1], device=x.device).unsqueeze(-1)
            tangent = TangentTensor(data=torch.cat([time, space], dim=x.man_dim), manifold=self, man_dim=x.man_dim)
            return self.expmap(tangent)
            # return space
        elif impl == "naive" or impl == "correction":
            space = x.tensor @ z.tensor.T
            if bias is not None:
                space += bias
            time = torch.sqrt(torch.norm(space, p=2, dim=x.man_dim, keepdim=True) ** 2 + 1.0 / self.c())
            return ManifoldTensor(torch.cat([time, space], dim=x.man_dim), manifold=self, man_dim=x.man_dim)
        elif impl == "chen":
            vT = z.tensor[0]
            W = z.tensor[1:]
            bias = bias.squeeze()
            b_0 = bias[0]
            eps = (1. / self.c()).sqrt() + 1e-6
            time = l * F.sigmoid(x.tensor @ vT + b_0) + eps
            num = (time ** 2 - 1. / self.c()).sqrt()
            space_unnormalised = F.relu(dropout(x.tensor)) @ W.T
            space = num.unsqueeze(x.man_dim) / space_unnormalised.norm(dim=x.man_dim, keepdim=True) * space_unnormalised
            return ManifoldTensor(torch.cat([time.unsqueeze(dim=x.man_dim), space], dim=x.man_dim), manifold=self, man_dim=x.man_dim)

    def multiheadattention(self,
                           query: ManifoldTensor,
                           key: ManifoldTensor,
                           value: ManifoldTensor,
                           att_module: nn.Module,
                           impl: str,
                           key_padding_mask: Optional[Tensor] = None,
                           attn_mask: Optional[Tensor] = None):
        if impl == "tangent":
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            # query_tangent = self.logmap(y=query)
            # key_tangent = self.logmap(y=key)
            # value_tangent = self.logmap(y=value)
            # attn_output = torch.zeros_like(query_tangent.tensor)
            attn_output, attn_output_weights = att_module(query=query[..., 1:],
                                                                   key=key[..., 1:],
                                                                   value=value[..., 1:],
                                                                   key_padding_mask=key_padding_mask,
                                                                   attn_mask=attn_mask)

            # output_tangent = TangentTensor(attn_output, manifold=self, man_dim=query.man_dim)
            # output = self.expmap(output_tangent)
            return attn_output
        elif impl == "naive" or impl == "correction":
            # The "naive" implementation follows the algorithm outlined in
            # "Fully Hyperbolic Neural Networks" by Chen et al. (2022).
            # It computes attention weights based on squared Lorentzian distance
            # and aggregates values by finding the Lorentz centroid.

            # Get the spatial dimension 'n' for the scaling factor, as per eq. [cite_start](5)[cite: 153].
            n = query.shape[-1] - 1

            # Calculate squared Lorentzian distance, as defined in the paper just before eq. [cite_start](4)[cite: 151].
            # This is used to compute the attention scores.
            sq_dist = squared_lorentzian_distance(query.tensor, key.tensor, self.c())

            # [cite_start]The attention score is the negative squared distance, scaled by sqrt(n)[cite: 153].
            attn_scores = -sq_dist / (n ** 0.5)

            # Apply attention mask if provided.
            if attn_mask is not None:
                attn_scores += attn_mask

            # Apply key padding mask if provided.
            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(1),
                    float("-inf"),
                )

            # Calculate attention weights using softmax, as per eq. [cite_start](5)[cite: 153].
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Aggregate the value vectors by computing the Lorentz centroid, as per eq. [cite_start](4)[cite: 165].
            # The `midpoint` function with weights performs this weighted aggregation and projection.
            output_tensor = midpoint(
                x=value.tensor,
                c=self.c(),
                w=attn_weights,
                dim=None  # Use weighted aggregation mode
            )

            # Wrap the resulting tensor in a ManifoldTensor.
            output = ManifoldTensor(
                data=output_tensor,
                manifold=self,
                man_dim=query.man_dim
            )

            return output

    def midpoint(
        self,
        x: ManifoldTensor,
        w: Optional[Tensor] = None,
        dim: Optional[int] = None,
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
        mu = midpoint(
            x=x.tensor,
            c=self.c(),
            w=w,
            dim=dim,
        )  # (B, H, L, (D / H) + 1)
        mu_space = mu[..., 1:].reshape(mu.shape[0], mu.shape[-2], -1)  # (B, L, D)
        mu_time = torch.sqrt(
            torch.norm(mu_space, dim=-1) ** 2 + 1 / self.c()
        ).unsqueeze(-1)  # (B, L, 1)
        mu = torch.cat([mu_time, mu_space], dim=-1)  # (B, L, D + 1)
        man_dim = x.man_dim - 1
        return ManifoldTensor(data=mu, manifold=self, man_dim=man_dim)

    def inner(self, u, v, dim=-1, keepdim=False, safe_mode=False):
        return self.minkowski_dot(u.tensor, v.tensor, dim=dim, keepdim=keepdim)

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
        impl: str,
        dim: int = 0,
    ) -> ManifoldTensor:
        check_if_man_dims_match(manifold_tensors)
        man_dim = manifold_tensors[0].man_dim

        # Resolve negative dim to its positive index equivalent.
        ndim = manifold_tensors[0].tensor.ndim
        resolved_dim = dim if dim >= 0 else ndim + dim

        # --- Case 1: Concatenating along the manifold dimension ---
        if resolved_dim == man_dim:
            if impl == "naive" or impl == "correction":
                # In the naive case, we concatenate the spatial dimensions and
                # calculate a new time dimension to ensure the result is on the manifold.
                tensors = [t.tensor for t in manifold_tensors]

                # Extract and concatenate all spatial parts of the tensors.
                # Assumes manifold dimension is the last one.
                spatial_parts = [t.narrow(man_dim, 1, t.shape[man_dim] - 1) for t in tensors]
                concatenated_spatial = torch.cat(spatial_parts, dim=man_dim)

                # Calculate the required time dimension for the new concatenated vector.
                # t = sqrt(||s||^2 + 1/c)
                time_squared = torch.sum(torch.pow(concatenated_spatial, 2), dim=man_dim, keepdim=True) + (
                            1.0 / self.c())
                new_time = torch.sqrt(time_squared)

                # Combine the new time dimension with the concatenated spatial part.
                final_tensor = torch.cat([new_time, concatenated_spatial], dim=man_dim)
                return ManifoldTensor(data=final_tensor, manifold=self, man_dim=man_dim)

            elif impl == "tangent":
                # In the tangent case, we map to the tangent space, perform a
                # standard Euclidean concatenation, and map back.
                tangent_tensors = [self.logmap(None, t) for t in manifold_tensors]
                concatenated_tangent_data = torch.cat([t.tensor for t in tangent_tensors], dim=man_dim)

                final_tangent = TangentTensor(data=concatenated_tangent_data, manifold=self, man_dim=man_dim)
                return self.expmap(final_tangent)

            else:
                raise ValueError(f"Unknown implementation for cat: {impl}")

        # --- Case 2: Concatenating along a non-manifold dimension ---
        else:
            # If concatenating along any other dimension, the operation is a simple
            # torch.cat, as it doesn't affect the manifold's structure.
            cat_tensor = torch.cat([t.tensor for t in manifold_tensors], dim=dim)
            return ManifoldTensor(data=cat_tensor, manifold=self, man_dim=man_dim)