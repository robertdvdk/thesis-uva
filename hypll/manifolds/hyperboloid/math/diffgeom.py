import torch

from .linalg import lorentz_dot


def expmap0(v: torch.Tensor, c: torch.Tensor, man_dim: int) -> torch.Tensor:
    """
    Calculates the exponential map at the origin, in the direction of tangent vector v, with curvature c.
    For definition, see #TODO.

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector.
    c : torch.Tensor
        The negative of the curvature of the hyperboloid in which the tensors lie (by GeoOpt convention). Note that
        this will be a positive value, as the curvature of hyperbolic space is negative.
    man_dim : int
        Dimension along which tensor lies on the manifold.

    Returns
    -------
    torch.Tensor
        The exponential map at the origin of v.
    """
    v = v.transpose(dim0=man_dim, dim1=-1)
    time = torch.cosh(v.norm(dim=-1)).unsqueeze(-1)
    space = (
        torch.divide(torch.sinh(v.norm(dim=-1)), (v.norm(dim=-1) + 1e-5)).unsqueeze(-1)
        * v
    )
    return torch.cat((time, space), dim=man_dim)


def expmap(
    x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, man_dim: int
) -> torch.Tensor:
    """
    Calculates the exponential map at the origin, in the direction of tangent vector v, with curvature c.
    For definition, see #TODO.

    Parameters
    ----------
    x : torch.Tensor
        Point at which to calculate the exponential map.
    v : torch.Tensor
        Tangent vector.
    c : torch.Tensor
        The negative of the curvature of the hyperboloid in which the tensors lie (by GeoOpt convention). Note that
        this will be a positive value, as the curvature of hyperbolic space is negative.
    man_dim : int
        Dimension along which tensor lies on the manifold.

    Returns
    -------
    torch.Tensor
        The exponential map at the origin of v.
    """
    v_norm_c_sqrt = v.norm(dim=-1, keepdim=True) * c.sqrt()
    return (
        torch.cosh(v_norm_c_sqrt) * x
        + torch.divide(torch.sinh(v_norm_c_sqrt), v_norm_c_sqrt) * v
    )


def logmap0(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x = torch.zeros_like(y)
    x[..., 0] = 1.0
    beta = (lorentz_dot(y, x) * -c).diag().unsqueeze(-1)
    return torch.arccosh(beta) / (beta**2 - 1).sqrt() * (y - beta * x)


def logmap(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    beta = lorentz_dot(y, x) * -c
    return torch.arccosh(beta) / (beta**2 - 1).sqrt() * (y - beta * x)


def dist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return cdist(x.flatten(), y.flatten(), c)


def cdist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    return (
        1
        / c.sqrt()
        * torch.arccosh(torch.clamp(-c * lorentz_dot(x, y, c, dim=dim), min=1.0 + 1e-6))
    )


def project(x, c):
    space = x[..., 1:]
    time = torch.sqrt(torch.norm(space, dim=-1) ** 2 + 1 / c).unsqueeze(-1)
    proj = torch.cat((time, space), dim=-1)
    return proj


def euc_to_tangent(
    x: torch.Tensor,
    u: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    # Compute the Minkowski inner product ⟨x, u⟩_L
    minkowski_dot = -x[..., 0] * u[..., 0] + torch.sum(
        x[..., 1:] * u[..., 1:], dim=-1, keepdim=True
    )

    # Project u onto the tangent space by subtracting the component along x
    tangent_u = u + minkowski_dot * x

    return tangent_u
