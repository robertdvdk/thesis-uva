import torch
import torchvision
from hypll.manifolds.hyperboloid import Hyperboloid, Curvature
import numpy as np

from hypll.optim import RiemannianAdam
from model import HViT, ViT
import torchvision.transforms.v2 as v2
import os
import wandb
import json
import argparse
import random


def log_metrics(
    metrics: np.ndarray,
    epoch: int,
    step: int,
    split: str,
    batches_per_epoch: int,
    run: wandb.sdk.wandb_run.Run,
) -> np.ndarray:
    acc_avg = 100 * metrics[1] / metrics[0]
    loss_avg = metrics[2] / metrics[0]

    measure = "step" if split == "Train" else "epoch"
    measure_value = epoch * batches_per_epoch + step if split == "Train" else epoch

    run.log(
        {
            f"{split}/total_loss": loss_avg,
            measure: measure_value,
        }
    )

    run.log(
        {
            f"{split}/accuracy": acc_avg,
            measure: measure_value,
        }
    )

    metrics = np.zeros(3)

    return metrics


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    loss_fn: torch.nn.Module,
    run: wandb.sdk.wandb_run.Run,
):
    param_to_name = {param: name for name, param in model.named_parameters()}
    model.train()
    metrics = np.zeros(3)
    for step, batch in enumerate(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        pred = model.forward(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step(param_to_name)
        optimizer.zero_grad()
        batch_size = x.size(0)
        metrics[0] += batch_size
        metrics[1] += (pred.argmax(dim=1) == y).sum().item()
        metrics[2] += loss.item() * batch_size

        if step % 10 == 0 and run:
            metrics = log_metrics(metrics, epoch, step, "Train", len(train_loader), run)

    return model


def evaluate_epoch(val_loader, device, model, epoch, loss_fn, run):
    model.eval()
    metrics = np.zeros(3)
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model.forward(x)
            loss = loss_fn(pred, y)

            batch_size = x.size(0)
            metrics[0] += batch_size
            metrics[1] += (pred.argmax(dim=1) == y).sum().item()
            metrics[2] += loss.item() * batch_size

        if run:
            log_metrics(metrics, epoch, batch, "Val", len(val_loader), run)
    return


def train_model(model, device, lr, epochs, train_loader, val_loader, run):
    optimizer = RiemannianAdam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model = train_epoch(model, optimizer, train_loader, device, epoch, loss_fn, run)
        evaluate_epoch(val_loader, device, model, epoch, loss_fn, run)
        torch.save(model.state_dict(), f"./model.pt")


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--root", type=str, default="~/data", help="Dataset root directory"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument("--heads", type=int, default=1, help="Number of heads")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--name", required=True, type=str, help="Wandb name")
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of blocks in the HViT"
    )
    parser.add_argument(
        "--use_pos_enc", action="store_true", help="Whether to use positional encoding"
    )
    parser.add_argument(
        "--manifold",
        required=True,
        type=str,
        help="Which manifold to create the model in",
    )

    args = parser.parse_args()

    # Dump to JSON
    os.makedirs("./runs", exist_ok=True)
    with open(f"./runs/{args.name}.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.use_wandb:
        wandb.login(key="1ae2d8a74e3f80102f0b5f15002a131ab3c42809")
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="thesis",
            name=args.name,
        )

        run.define_metric("step")
        run.define_metric("epoch")
        run.define_metric("Train/total_loss", step_metric="step")
        run.define_metric("Train/accuracy", step_metric="step")

        run.define_metric("Val/total_loss", step_metric="epoch")
        run.define_metric("Val/accuracy", step_metric="epoch")
    else:
        run = None

    assert args.dataset in ["cifar10", "cifar100"]
    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.root, "cifar10"),
            download=True,
            train=True,
            transform=v2.Compose(
                [
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            ),
        )
        valset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.root, "cifar10"),
            download=True,
            train=False,
            transform=v2.Compose(
                [
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            ),
        )
        num_classes = 10
        img_size = 32
        channels = 3
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=os.path.join(args.root, "cifar100"),
            download=True,
            train=True,
            transform=v2.Compose(
                [
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            ),
        )
        valset = torchvision.datasets.CIFAR100(
            root=os.path.join(args.root, "cifar100"),
            download=True,
            train=False,
            transform=v2.Compose(
                [
                    v2.ToTensor(),
                    v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            ),
        )
        num_classes = 100
        img_size = 32
        channels = 3

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    assert args.manifold in ["Hyperbolic", "Euclidean"]
    if args.manifold == "Hyperbolic":
        manifold = Hyperboloid(Curvature(value=np.log(np.exp(1) - 1)))
        model = HViT(
            img_size=img_size,
            patch_size=args.patch_size,
            in_channels=channels,
            embed_dim=args.embed_dim,
            num_heads=args.heads,
            num_classes=num_classes,
            num_layers=args.num_layers,
            use_pos_enc=args.use_pos_enc,
            manifold=manifold,
        )
    else:
        model = ViT(
            img_size=img_size,
            patch_size=args.patch_size,
            in_channels=channels,
            embed_dim=args.embed_dim,
            num_heads=args.heads,
            num_classes=num_classes,
            num_layers=args.num_layers,
            use_pos_enc=args.use_pos_enc,
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_model(
        model,
        device,
        lr=args.lr,
        epochs=args.epochs,
        train_loader=trainloader,
        val_loader=valloader,
        run=run,
    )
