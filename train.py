"""
Train a flow matching generative model on MNIST.

Based on "An Introduction to Flow Matching and Diffusion Models" (arxiv 2506.02070)

Flow matching learns a vector field u_t(x) that transports samples from
a simple prior p_0 = N(0, I) to the data distribution p_1 = p_data.

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


def sample_and_save(model, device, epoch, args):
    """Generate a grid of samples and save as image."""
    images = sample(model, device, n_samples=100, n_steps=args.sample_steps)
    images = images.cpu()

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i][j].imshow(images[i * 10 + j, 0], cmap='gray', vmin=0, vmax=1)
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

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    Dataset = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    dataset = Dataset(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = UNet(base_channels=args.channels).to(device)
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
            sample_and_save(model, device, epoch, args)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print(f"Model saved to {args.output_dir}/model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching on MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()
    train(args)
