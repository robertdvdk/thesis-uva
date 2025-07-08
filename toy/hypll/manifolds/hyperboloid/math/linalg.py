from typing import Optional
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


# ... (rest of linalg.py remains the same) ...
def lorentz_fully_connected(
        x: torch.Tensor,
        W: torch.Tensor,
        bias: Optional[torch.Tensor],
        c: torch.Tensor,
        num_heads: int = 1,
) -> torch.Tensor:
    """
    Applies a Lorentzian fully connected layer.

    This layer performs a standard matrix multiplication on an input tensor from the hyperboloid,
    and then projects the result back. This is a specific convention used in some models.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor on the hyperboloid of shape (..., D_in + 1).
    W : torch.Tensor
        Weight matrix of shape (D_out, D_in + 1).
    bias : Optional[torch.Tensor]
        Bias vector of shape (D_out).
    c : torch.Tensor
        Positive curvature constant of the manifold.
    num_heads : int, optional
        If greater than 1, reshapes the output for multi-head attention, by default 1.

    Returns
    -------
    torch.Tensor
        The output tensor on the hyperboloid.
    """
    space = x @ W.T
    if bias is not None:
        space = space + bias

    if num_heads > 1:
        space = space.view(*space.shape[:-1], num_heads, -1)

    # Compute time coordinate to ensure the point is on the hyperboloid
    time = torch.sqrt(torch.norm(space, p=2, dim=-1, keepdim=True) ** 2 + 1.0 / c)
    return torch.cat([time, space], dim=-1)


def lorentz_patch_embedding(
        x: torch.Tensor,
        weights: torch.Tensor,
        positional_encoding: torch.Tensor,
        c: torch.Tensor,
        patch_size: int,
) -> torch.Tensor:
    """
    Performs patch embedding for vision transformers on the hyperboloid.

    This function first flattens image patches to Euclidean vectors, then maps them
    to the unit hyperboloid by prepending a time coordinate. A linear transformation
    is applied in the ambient space, and the result is projected back to the
    hyperboloid with curvature `c`.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor of shape (B, C, H, W).
    weights : torch.Tensor
        Weight matrix for the linear projection, shape (D_out, C * patch_size**2 + 1).
    positional_encoding : torch.Tensor
        Positional encodings to be added to the spatial components.
    c : torch.Tensor
        Positive curvature constant of the manifold.
    patch_size : int
        The size of each square patch.

    Returns
    -------
    torch.Tensor
        The resulting patch embeddings on the hyperboloid.
    """
    # Unfold image into patches and flatten
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.contiguous().view(x.shape[0], x.shape[1], -1, patch_size * patch_size)
    x = x.permute(0, 2, 1, 3).flatten(2)

    # Prepend time coordinate to project the patch vector to the unit hyperboloid
    time = torch.sqrt(torch.norm(x, p=2, dim=-1, keepdim=True) ** 2 + 1.0)
    x_h = torch.cat([time, x], dim=-1)

    # Apply linear transformation in ambient space.
    # The result is considered the new spatial component.
    space = x_h @ weights.T + positional_encoding

    # Compute the new time coordinate to project back to the target hyperboloid
    new_time = torch.sqrt(torch.norm(space, p=2, dim=-1, keepdim=True) ** 2 + 1.0 / c)
    return torch.cat([new_time, space], dim=-1)