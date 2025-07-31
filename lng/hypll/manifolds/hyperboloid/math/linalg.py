import torch

def lorentz_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Lorentzian dot product between two tensors using optimized
    batched matrix multiplication to avoid creating large intermediate tensors.

    This function handles both element-wise dot products (if x and y are
    individual vectors) and pairwise dot product matrices for attention.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor. Shape: `(..., N, D+1)`.
    y : torch.Tensor
        The second tensor. Shape: `(..., M, D+1)`.

    Returns
    -------
    torch.Tensor
        The Lorentzian dot product. Shape: `(..., N, M)`.
    """
    # Decompose tensors into time and space components.
    x_time, x_space = x[..., 0:1], x[..., 1:]
    y_time, y_space = y[..., 0:1], y[..., 1:]

    # Use batched matrix multiplication (@) for the spatial component.
    # This is highly optimized and memory-efficient.
    # (B, N, D) @ (B, M, D).transpose(-2, -1) -> (B, N, D) @ (B, D, M) -> (B, N, M)
    space_dots = x_space @ y_space.transpose(-2, -1)

    # Use the same operation for the time component.
    time_dots = - (x_time @ y_time.transpose(-2, -1))

    # The result is the full pairwise dot product matrix.
    return time_dots + space_dots



def squared_lorentzian_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared Lorentzian distance between two sets of points.

    The squared distance is given by: $d(x, y)^2 = -2/c - 2 * <x, y>_L$

    Parameters
    ----------
    x : torch.Tensor
        First set of points on the hyperboloid.
    y : torch.Tensor
        Second set of points on the hyperboloid.
    c : torch.Tensor
        Positive curvature constant of the manifold.

    Returns
    -------
    torch.Tensor
        The squared Lorentzian distance between x and y.
    """
    return -2.0 / c - 2.0 * lorentz_dot(x, y)