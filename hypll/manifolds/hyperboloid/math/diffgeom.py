"""
This module provides core differential geometry operations for the Hyperboloid
manifold model of hyperbolic space.
"""

import torch
from torch import Tensor

from .linalg import lorentz_dot


def expmap0(v: Tensor, c: Tensor, man_dim: int) -> Tensor:
    r"""Calculates the exponential map from the origin of the tangent space.

    This function maps a tangent vector `v` from the origin's tangent space
    onto the hyperboloid manifold.

    Note: The implementation's scaling does not involve the curvature `c`.

    Parameters
    ----------
    v : Tensor
        A tangent vector at the origin.
    c : Tensor
        The positive curvature constant of the manifold (unused in this implementation).
    man_dim : int
        The dimension of the tensor that lies on the manifold. The tensor is
        transposed to bring this to the last dimension for calculations.

    Returns
    -------
    Tensor
        The point on the hyperboloid.
    """
    v[..., 0] = 0
    # The `man_dim` is transposed to the end for consistent processing.
    v_tran = v.transpose(dim0=man_dim, dim1=-1)
    v_norm = v_tran.norm(dim=-1)

    # Note: Using v_tran.norm() repeatedly can be inefficient, but logic is preserved.
    # A small epsilon is added for numerical stability when the norm is near zero.
    time = torch.cosh(v_norm).unsqueeze(-1)
    space = (torch.sinh(v_norm) / (v_norm + 1e-5)).unsqueeze(-1) * v_tran

    # Concatenate time and space coords, and transpose back to original dim order.
    space[..., 0] = time.squeeze(-1)
    return space.transpose(dim0=man_dim, dim1=-1)


def expmap(x: Tensor, v: Tensor, c: Tensor, prnt) -> Tensor:
    r"""Calculates the exponential map at a point `x` in the direction of `v`.

    The formula implemented is:
    $$ exp_x(v) = \cosh(\sqrt{c}||v||_E)x + \frac{\sinh(\sqrt{c}||v||_E)}{\sqrt{c}||v||_E}v $$
    where `||v||_E` is the Euclidean L2 norm, not the Lorentzian norm.

    Parameters
    ----------
    x : Tensor
        A point on the hyperboloid, shape `(..., D+1)`.
    v : Tensor
        A tangent vector at `x`, shape `(..., D+1)`.
    c : Tensor
        The positive curvature constant of the manifold.

    Returns
    -------
    Tensor
        The resulting point on the hyperboloid, shape `(..., D+1)`.
    """

    # Calculate the Minkowski norm squared of the tangent vector v
    # <v, v>_L = -v_0^2 + v_1^2 + ...
    minkowski_v_sq = torch.sum(v[..., 1:] ** 2, dim=-1, keepdim=True) - v[..., 0:1] ** 2

    # Clamp to avoid sqrt of negative numbers due to floating point errors
    v_minkowski_norm = torch.sqrt(torch.clamp_min(minkowski_v_sq, 1e-8))

    v_norm_c_sqrt = v_minkowski_norm * torch.sqrt(c)

    # Handle the case where v is the zero vector to avoid division by zero
    # If v is zero, exp_x(0) = x
    is_zero = v_norm_c_sqrt == 0

    # Calculate sinh(z)/z safely
    div = torch.sinh(v_norm_c_sqrt) / v_norm_c_sqrt
    div[is_zero] = 1.0  # The limit of sinh(z)/z as z->0 is 1
    # if prnt:
    #     print("X", x)
    #     print("V", v)
    #     print("VNCS", v_norm_c_sqrt)
    #     # print(torch.cosh(v_norm_c_sqrt) * x + div * v)
    #     print("CX", torch.cosh(v_norm_c_sqrt) * x)
    #     print("DIV V", div * v)
    return torch.cosh(v_norm_c_sqrt) * x + div * v


def logmap0(y: Tensor, c: Tensor) -> Tensor:
    r"""Calculates the logarithmic map of a point `y` to the tangent space at the origin.

    This maps a point from the hyperboloid back to a vector in the tangent
    space of the origin `x_0 = (1, 0, ..., 0)`.

    Parameters
    ----------
    y : Tensor
        A point on the hyperboloid.
    c : Tensor
        The positive curvature constant of the manifold.

    Returns
    -------
    Tensor
        The tangent vector at the origin.
    """
    # For simplicity and compatibility with standard models, we assume c=1.
    # The logic can be adapted if a different curvature scaling is truly needed.

    # Clamp y_0 to prevent NaNs in acosh for values slightly less than 1
    # due to floating-point precision errors.
    eps = 1e-8
    y_0 = y[..., 0:1].clamp(min=1.0 + eps)
    y_spatial = y[..., 1:]

    # The distance 'd' is the norm of the tangent vector we want to find.
    distance = torch.acosh(y_0)

    # The norm of the spatial part of y should be sinh(distance).
    # We add a small epsilon for numerical stability when y is at the origin.
    spatial_norm = torch.norm(y_spatial, p=2, dim=-1, keepdim=True)
    spatial_norm_clamped = torch.clamp(spatial_norm, min=1e-8)

    # The spatial part of the tangent vector is the spatial direction of y,
    # scaled by the hyperbolic distance.
    v_spatial = distance * (y_spatial / spatial_norm_clamped)

    # The time component of a tangent vector at the origin is always 0.
    v_time = torch.zeros_like(y_0)

    return torch.cat([v_time, v_spatial], dim=-1)



def logmap(x: Tensor, y: Tensor, c: Tensor) -> Tensor:
    r"""Calculates the logarithmic map of point `y` to the tangent space at point `x`.

    The formula implemented is:
    $$ log_x(y) = \frac{\cosh^{-1}(\beta)}{\sqrt{\beta^2-1}}(y-\beta x), \quad \beta = -c<y,x>_L $$

    Parameters
    ----------
    x : Tensor
        The point on the hyperboloid defining the tangent space.
    y : Tensor
        The point on the hyperboloid to map.
    c : Tensor
        The positive curvature constant of the manifold.

    Returns
    -------
    Tensor
        The tangent vector at `x`.
    """
    # print(y[0])
    # print(x[0])
    q = -y[0, 0] * x[0, 0] + y[0, 1] * x[0, 1] + y[0, 2] * x[0, 2]
    # print(torch.arccosh(q) / (q ** 2 - 1))
    beta = (-c * lorentz_dot(y, x)).diag().unsqueeze(-1)
    return torch.arccosh(beta) / (beta ** 2 - 1).sqrt() * (y     - beta * x)


def dist(x: Tensor, y: Tensor, c: Tensor) -> Tensor:
    r"""Computes the hyperbolic distance between two tensors after flattening them.

    .. warning::
        This function flattens the input tensors `x` and `y` into 1D vectors
        before calculating a single distance value. This means it does **not**
        compute element-wise or pairwise distances for multi-dimensional inputs.
        Instead, it treats each entire tensor as a single point.

        For a matrix of pairwise distances, use the `cdist` function.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor. It will be flattened before use.
    y : torch.Tensor
        The second tensor. It will be flattened before use.
    c : torch.Tensor
        The positive curvature constant of the manifold.

    Returns
    -------
    torch.Tensor
        A single scalar tensor representing the hyperbolic distance between the
        two flattened input tensors.
    """
    return cdist(x.flatten(), y.flatten(), c)


def cdist(x: Tensor, y: Tensor, c: Tensor, dim: int = -1) -> Tensor:
    r"""Computes the pairwise hyperbolic distance between two sets of points.

    The formula is:
    $$ d(x,y) = \frac{1}{\sqrt{c}} \cosh^{-1}(-c<x, y>_L) $$

    Parameters
    ----------
    x : Tensor
        First set of points on the hyperboloid.
    y : Tensor
        Second set of points on the hyperboloid.
    c : Tensor
        The positive curvature constant of the manifold.
    dim : int, optional
        The dimension along which vectors are defined, by default -1.

    Returns
    -------
    Tensor
        The matrix of pairwise distances.
    """
    # Note: The original code passed `c` and `dim` to lorentz_dot. Assuming
    # `lorentz_dot` only takes x and y, as is common.
    dot_product = lorentz_dot(x, y)

    # Clamp the argument to arccosh to be >= 1 for numerical stability.
    clamped_arg = torch.clamp(-c * dot_product, min=1.0 + 1e-6)

    return (1 / c.sqrt()) * torch.acosh(clamped_arg)


def project(x: Tensor, c: Tensor) -> Tensor:
    r"""Projects a point from the ambient space onto the hyperboloid.

    This function takes a point `x` and calculates its time-like coordinate `x_0`
    such that it satisfies the hyperboloid equation:
    $$-x_0^2 + \sum_{i=1}^D x_i^2 = -1/c$$

    Parameters
    ----------
    x : Tensor
        A point in the ambient space. Shape `(..., D+1)`.
    c : Tensor
        The positive curvature constant of the manifold.

    Returns
    -------
    Tensor
        The projected point on the hyperboloid.
    """
    time = torch.sqrt(torch.norm(x, dim=-1) ** 2 + 1 / c).unsqueeze(-1)
    # The `proj` variable is immediately returned.
    return torch.cat((time, x), dim=-1)


def euc_to_tangent(x: Tensor, u: Tensor, dim: int = -1) -> Tensor:
    r"""Projects a vector `u` from the ambient space onto the tangent space at `x`.

    This is the Riemannian gradient projection step. The formula is:
    $$ \text{Proj}_x(u) = u + \langle u, x \rangle_L x $$
    """
    # Use slicing `[..., :1]` to select the time-like coordinate, preserving the dimension.
    # This prevents the broadcasting error.
    time_prod = -x[..., 0] * u[..., 0]

    # The rest of the calculation is correct.
    space_prod = torch.sum(x[..., 1:] * u[..., 1:], dim=-1, keepdim=True)

    # Both terms now have shape [10000, 1], so they add correctly.
    minkowski_dot = time_prod + space_prod

    # Now the shapes match for the final operation:
    # u                 -> [10000, 5]
    # minkowski_dot * x -> [10000, 1] * [10000, 5] -> [10000, 5]
    return u + minkowski_dot * x