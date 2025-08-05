from typing import Optional

import torch

from .linalg import lorentz_dot


def midpoint(
        x: torch.Tensor,
        c: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Finds the Fréchet mean (midpoint) of points on the Hyperboloid manifold.

    This function operates in two modes:
    1.  **Weighted Aggregation (dim=None):** If `dim` is not provided, this function
        performs a rank-preserving weighted aggregation using a weight matrix `w`.
        This is primarily used for the attention mechanism.
        - `x` shape: `(..., L, D+1)`
        - `w` shape: `(..., L, L)`
        - Output shape: `(..., L, D+1)`

    2.  **Geometric Mean (dim is specified):** If a `dim` is provided, this function
        computes the geometric mean of points along that dimension, reducing the
        tensor's rank. This is used for pooling or residual connections.
        - `x` shape: `(..., k, ..., D+1)` where `k` is the size of `dim`.
        - `w` shape (if provided): `(k,)`
        - Output shape: `(..., ..., D+1)` with the `dim` dimension removed.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of input points on the hyperboloid.
    c : torch.Tensor
        The positive curvature constant of the manifold.
    w : Optional[torch.Tensor], optional
        A tensor of weights. Its shape determines the mode of operation.
        If `None`, uniform weights are used for geometric mean calculation.
    dim : Optional[int], optional
        The dimension to average over for a geometric mean. If `None`, the function
        operates in weighted aggregation mode.

    Returns
    -------
    torch.Tensor
        The resulting midpoint(s) on the hyperboloid.

    Raises
    ------
    ValueError
        If arguments are provided in an invalid combination.
    """
    if dim is not None:
        # --- Mode 2: Geometric Mean (Rank-Reducing) ---
        if w is None:
            # If no weights are given, perform a simple unweighted mean.
            numerator = torch.mean(x, dim=dim)
        else:
            # Perform a weighted mean along the specified dimension.
            # Prepare weights for broadcasting.
            w_shape = [1] * x.ndim
            w_shape[dim] = w.shape[0] if w.ndim == 1 else w.shape[dim]
            w = w.view(w_shape)
            numerator = torch.sum(w * x, dim=dim)
    elif w is not None:
        # --- Mode 1: Weighted Aggregation (Rank-Preserving) ---
        # This is the attention use case.
        numerator = w @ x
    else:
        raise ValueError(
            "Invalid arguments. Either `w` must be provided for weighted aggregation, "
            "or `dim` must be specified for a geometric mean."
        )

    # --- Common Projection Logic ---
    # The following projects the result from the ambient space back onto the hyperboloid.

    # 1. Compute the squared Minkowski norm for each resulting vector.
    # lorentz_dot(v, v) returns a matrix of pairwise products; the diagonal contains the norms <v_i, v_i>.
    dot_prod_matrix = lorentz_dot(numerator, numerator)
    if numerator.ndim > 1:
        diag = torch.einsum("...ii -> ...i", dot_prod_matrix)
    else:  # The result is a single vector, so the dot product is a scalar.
        diag = dot_prod_matrix

    # 2. Calculate the denominator for projection: sqrt(c) * ||numerator||_L
    # Add epsilon for numerical stability.
    clamped_diag = torch.clamp(diag.abs(), min=1e-12)
    denominator = (c.sqrt() * clamped_diag.sqrt()).unsqueeze(-1) + 1e-8

    # 3. Perform the projection.
    midpoint = numerator / denominator
    return midpoint