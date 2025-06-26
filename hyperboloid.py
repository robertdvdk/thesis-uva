import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("tkagg")
# Define a grid for the parameters
u = np.linspace(0, 2 * np.pi, 100)  # Angle around the z-axis
v = np.linspace(0.1, 2, 50)  # Height along the hyperboloid

# Parametric equations for the hyperboloid
U, V = np.meshgrid(u, v)
X = np.sinh(V) * np.cos(U)
Y = np.sinh(V) * np.sin(U)
Z = np.cosh(V)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Draw the hyperboloid surface
ax.plot_surface(Z, X, Y, alpha=0.7, cmap="viridis", edgecolor="none")

a = torch.tensor([torch.tensor(3).sqrt(), torch.tensor(1), torch.tensor(1)])
b = torch.tensor([0, -torch.tensor(2).sqrt() / 2, torch.tensor(2).sqrt() / 2])
c = torch.tensor(
    [torch.tensor(2).sqrt(), torch.tensor(6).sqrt() / 2, torch.tensor(6).sqrt() / 2]
)


ax.quiver(
    0, 0, 0, a[0], a[1], a[2], color="red", linewidth=1, label="Vector (2, √1.5, √1.5)"
)
ax.quiver(
    0, 0, 0, b[0], b[1], b[2], color="blue", linewidth=1, label="Vector (2, √1.5, √1.5)"
)
ax.quiver(
    0,
    0,
    0,
    c[0],
    c[1],
    c[2],
    color="green",
    linewidth=1,
    label="Vector (2, √1.5, √1.5)",
)

l1 = torch.stack([torch.cosh(t) * a + torch.sinh(t) * b for t in (torch.tensor(v) - 1)])
x_values1 = l1[:, 0]
y_values1 = l1[:, 1]
z_values1 = l1[:, 2]
ax.plot(
    x_values1,
    y_values1,
    z_values1,
    color="magenta",
    linewidth=2,
    label="Geodesic Curve",
)

l2 = torch.stack([torch.cosh(t) * a + torch.sinh(t) * c for t in (torch.tensor(v) - 1)])
x_values2 = l2[:, 0]
y_values2 = l2[:, 1]
z_values2 = l2[:, 2]
ax.plot(
    x_values2, y_values2, z_values2, color="black", linewidth=2, label="Geodesic Curve"
)

# Set labels and aspect ratio
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_zlabel("y")
ax.set_title("Hyperboloid Model of the Hyperbolic Plane")

plt.show()
