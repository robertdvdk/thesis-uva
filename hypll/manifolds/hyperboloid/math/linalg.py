from typing import Optional
import torch


def lorentz_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Takes in two tensors of shapes (..., N, D + 1) and (..., M, D + 1) and computes the
    Lorentzian dot product over the last dimension to return a tensor of shape (..., N, M).

    Assumes the first coordinate is the "time" component, with signature (-, +, +, ...).

    Parameters
    ----------
    x : torch.Tensor
        First tensor, shape (..., N, D + 1).
    y : torch.Tensor
        Second tensor, shape (..., M, D + 1).

    Returns
    -------
    torch.Tensor
        Lorentzian dot product of x and y, shape (..., N, M).
    """
    if len(x.shape) == 1 and len(y.shape) == 1:
        time_part = -x[0] * y[0]
        space_part = (x[1:] * y[1:]).sum()
        return time_part + space_part
    # Reshape to allow pairwise computation
    x_expanded = x.unsqueeze(-2)  # shape (..., a, 1, n)
    y_expanded = y.unsqueeze(-3)  # shape (..., 1, b, n)

    # Compute time component (negative sign) + spatial components
    time_part = -x_expanded[..., 0] * y_expanded[..., 0]
    space_part = torch.sum(x_expanded[..., 1:] * y_expanded[..., 1:], dim=-1)

    return time_part + space_part


def squared_lorentzian_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Takes in two 2-dimensional tensors of shapes (..., N, D + 1) and (..., M, D + 1) and computes the
    squared Lorentzian distance between them to return a tensor of shape (..., N, M).

    Parameters
    ----------
    x : torch.Tensor
        First tensor, shape (..., N, D + 1).
    y : torch.Tensor
        Second tensor, shape (..., M, D + 1).
    c : torch.Tensor
        The negative of the curvature of the hyperboloid in which the tensors lie (by GeoOpt convention). Note that
        this will be a positive value, as the curvature of hyperbolic space is negative.

    Returns
    -------
    torch.Tensor
        Squared Lorentzian distance between x and y, shape (..., N, M).
    """
    return -2 / c - 2 * lorentz_dot(x, y)


def lorentz_fully_connected(
    x: torch.Tensor,
    W: torch.Tensor,
    bias: Optional[torch.Tensor],
    c: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Calculates a Lorentz fully connected layer by applying a weight matrix on the input vector,
    whose output signifies the space coordinates. The corresponding time coordinate is then computed.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, shape (..., N, D + 1).
    W : torch.Tensor
        Weight tensor, shape (D, D + 1).
    bias : Optional[torch.Tensor]
        Bias vector, shape (D).
    c : torch.Tensor
        The negative of the curvature of the hyperboloid in which the tensors lie (by GeoOpt convention). Note that
        this will be a positive value, as the curvature of hyperbolic space is negative.
    act_fn : Callable
        Activation function.
    num_heads : int
        Used in case the fully connected layer is part of a multi-head attention mechanism.
    dim : int
        Dimension along which tensor lies on the manifold.

    Returns
    -------
    torch.Tensor
        Output tensor, shape (..., N, (H), D + 1), where H is the number of heads. The H dimension does not exist if num_heads is 0 or 1.
    """
    if bias is None:
        bias = torch.zeros(W.size(0))
    space = x @ W.T + bias.unsqueeze(0)
    # If we use multiple heads, reshape the space tensor
    # from (..., n, D) to (..., n, num_heads, D / num_heads).
    if num_heads > 1:
        space = space.view((*space.shape[:-1], num_heads, -1))
    time = torch.sqrt(torch.norm(space, dim=-1) ** 2 + 1 / c).unsqueeze(-1)
    output = torch.cat([time, space], dim=-1)
    return output


def lorentz_patch_embedding(
    x: torch.Tensor,
    weights: torch.Tensor,
    positional_encoding: torch.Tensor,
    c: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    B, C, H, W = x.shape
    x = x.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )  # (B, C, H // patch_size, W // patch_size, patch_size, patch_size
    x = x.contiguous().view(
        B, C, -1, patch_size * patch_size
    )  # (B, C, L, patch_size * patch_size)
    x = x.permute(0, 2, 1, 3).flatten(2)  # (B, L, C * patch_size * patch_size)
    time = torch.sqrt(torch.norm(x, dim=-1) ** 2 + 1).unsqueeze(-1)
    x = torch.cat([time, x], dim=-1)

    # Std of spatial coordinates is 1/n here, so variance is 1/n^2
    # This means that variance of time coordinates is approx. 1/4n.
    space = x @ weights.T
    space += positional_encoding
    # If we use multiple heads, reshape the space tensor
    # from (..., n, D) to (..., n, num_heads, D / num_heads).
    time = torch.sqrt(torch.norm(space, dim=-1) ** 2 + 1 / c).unsqueeze(-1)
    output = torch.cat([time, space], dim=-1)
    return output


def alternative_linear(normal, x):
    q = torch.asinh(lorentz_dot(normal, x) / lorentz_dot(normal, normal).sqrt())
    print(q)


alternative_linear(torch.tensor([0, -1, 5, 2]), torch.tensor([1, 0, 0, 0]))
