# debug.py


import torch
import torch.profiler

from hmodel import HToy
import numpy as np
from hypll.manifolds.hyperboloid import Hyperboloid, Curvature
from hypll.tensors import ManifoldTensor
import hypll.optim
import random
import sys

def tangent_space_mse_loss(y_pred_man, y_target_man):
    # Map both points from the manifold to the tangent space at the origin
    tangent_pred = y_pred_man.manifold.logmap(y_pred_man)
    tangent_target = y_target_man.manifold.logmap(y_target_man)

    # Calculate simple Euclidean MSE in the tangent space
    return torch.nn.functional.mse_loss(tangent_pred.tensor, tangent_target.tensor)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    manifold = Hyperboloid(Curvature(value=np.log(np.exp(1) - 1)))

    set_seed(42)
    d = 3

    N = torch.tensor(40)
    model = HToy(dim=d-1, manifold=manifold, impl="naive")
    x = torch.ones(d)
    x[0] = np.sqrt(d+1)
    y = torch.zeros(d)
    y[0] = torch.cosh(N)
    y[1] = torch.sinh(N)
    x_man = ManifoldTensor(data=x.repeat(64, 1), manifold=manifold)
    y_man = ManifoldTensor(data=y.repeat(64, 1), manifold=manifold)
    optimizer = hypll.optim.RiemannianAdam(model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-18, correction=True, weight_decay=0)
    param_to_name = {param: name for name, param in model.named_parameters()}
    for i in range(10000):
        print(i)
        y_pred = model(x_man)
        l = tangent_space_mse_loss(y_pred, y_man).mean()
        l.backward()
        print("LOSS:", l)
        # print("PREDICTED:", y_pred[0])
        # print("TRUE", y_man[0])

        if l < 0.001:
            print(i)
            print(l.item())
            print(y_pred.tensor[0])
            print(y_man.tensor[0])
            break
        optimizer.step(param_to_name)
        optimizer.zero_grad()
        if y_pred.tensor.isnan().any():
            sys.exit()
    for n, p in model.named_parameters():
        print(n, p)



if __name__ == "__main__":
    main()