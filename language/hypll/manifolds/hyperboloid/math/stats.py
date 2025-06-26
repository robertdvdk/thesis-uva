from typing import Optional

import torch

from .linalg import lorentz_dot

_TOLEPS = {torch.float32: 1e-6, torch.float64: 1e-12}


def midpoint(
    x: torch.Tensor,
    c: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Finds the point that minimises the (weighted) sum of squared distances to the input points in x.

    Parameters
    ----------
    x : torch.Tensor
        The input points, shape (B, (H), L, (D / H) + 1). (H) represents the number of heads, and does not exist if
        the input tensor does not come from a multi-head attention operation.
    c : torch.Tensor
        The negative of the curvature of the hyperboloid in which the tensors lie (by GeoOpt convention). Note that
        this will be a positive value, as the curvature of hyperbolic space is negative.
    man_dim : int, optional
        The dimension along which points lie on the manifold, by default -1
    midpoint_dim : Union[int, list[int]], optional
        The dimension along which to find the midpoint, by default 0
    w : Optional[torch.Tensor], optional
        Weights to use in the weighted sum, shape (B, (H), L, L), by default None.

    Returns
    -------
    torch.Tensor
        The midpoints.
    """
    if w is None:
        shape = (1,) + (x.shape[-2],)
        w = torch.ones(shape) / shape[0]
        w = w.to(x.device)
    numerator = w @ x
    diag = torch.einsum("...ii -> ...i", lorentz_dot(numerator, numerator))
    denominator = (c.sqrt() * diag.abs().sqrt()).unsqueeze(-1) + 1e-8
    midpoint = numerator / denominator
    return midpoint
