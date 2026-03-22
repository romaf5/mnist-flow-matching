"""
Train a flow matching generative model.

Based on "An Introduction to Flow Matching and Diffusion Models" (arxiv 2506.02070)

Supports MNIST, Fashion-MNIST, and CIFAR-10.

Conditional Optimal Transport path:
    x_t = t * z + (1 - t) * epsilon
    target = z - epsilon

Loss:
    L = E[|| u_theta(x_t, t) - (z - epsilon) ||^2]
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse

from model import UNet
from sample import sample


DATASET_CONFIG = {
    "mnist":   {"cls": datasets.MNIST,        "channels": 1, "size": 28},
    "fashion": {"cls": datasets.FashionMNIST, "channels": 1, "size": 28},
    "cifar10": {"cls": datasets.CIFAR10,      "channels": 3, "size": 32},
}


def get_transform(ds_cfg):
    if ds_cfg["channels"] == 1:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


def sample_and_save(model, device, epoch, args, ds_cfg):
    """Generate a grid of samples and save as image."""
    in_ch = ds_cfg["channels"]
    img_size = ds_cfg["size"]
    images = sample(model, device, n_samples=100, n_steps=args.sample_steps,
                    in_channels=in_ch, img_size=img_size)
    images = images.cpu()

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            img = images[i * 10 + j]
            if in_ch == 1:
                axes[i][j].imshow(img[0], cmap='gray', vmin=0, vmax=1)
            else:
                axes[i][j].imshow(img.permute(1, 2, 0).clamp(0, 1))
            axes[i][j].axis('off')
    plt.suptitle(f'Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    path = os.path.join(args.output_dir, f"samples_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved samples to {path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds_cfg = DATASET_CONFIG[args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)

    transform = get_transform(ds_cfg)
    dataset = ds_cfg["cls"](args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = UNet(in_channels=ds_cfg["channels"], base_channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for z, labels in loader:
            z = z.to(device)
            labels = labels.to(device)

            t = torch.rand(z.shape[0], device=device)
            epsilon = torch.randn_like(z)

            # CondOT interpolation
            t_expand = t[:, None, None, None]
            x_t = t_expand * z + (1 - t_expand) * epsilon
            target = z - epsilon

            # 10% label dropout for classifier-free guidance
            class_label = labels.clone()
            mask = torch.rand(z.shape[0], device=device) < 0.1
            class_label[mask] = 10

            pred = model(x_t, t, class_label)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample_and_save(model, device, epoch, args, ds_cfg)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"Model saved to {args.output_dir}/model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion", "cifar10"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()
    train(args)
