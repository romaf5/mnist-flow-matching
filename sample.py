"""
Sampling from a trained flow matching model.

Uses Euler integration of the learned vector field:
    x_{t+h} = x_t + h * u_theta(x_t, t)

Supports classifier-free guidance for class-conditional generation.
"""

import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse

from model import UNet


@torch.no_grad()
def sample(model, device, n_samples=64, n_steps=100, class_label=None, cfg_scale=2.0):
    """
    Generate images using Euler integration of the learned vector field.
    """
    model.eval()
    h = 1.0 / n_steps

    x = torch.randn(n_samples, 1, 28, 28, device=device)

    if class_label is not None:
        labels = torch.full((n_samples,), class_label, device=device, dtype=torch.long)
    else:
        labels = torch.arange(10, device=device).repeat(n_samples // 10 + 1)[:n_samples]

    uncond_labels = torch.full((n_samples,), 10, device=device, dtype=torch.long)

    t_val = 0.0
    for step in range(n_steps):
        t = torch.full((n_samples,), t_val, device=device)

        pred_cond = model(x, t, labels)
        pred_uncond = model(x, t, uncond_labels)
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        x = x + h * pred
        t_val += h

    x = x.clamp(-1, 1) * 0.5 + 0.5
    return x


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base_channels=args.channels).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt"),
                                     map_location=device, weights_only=True))

    images = sample(model, device, n_samples=args.n_samples,
                    n_steps=args.sample_steps, class_label=args.digit,
                    cfg_scale=args.cfg_scale)
    images = images.cpu()

    n = int(math.ceil(math.sqrt(args.n_samples)))
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            ax = axes[i][j] if n > 1 else axes
            if idx < args.n_samples:
                ax.imshow(images[idx, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "generated.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated {args.n_samples} images -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MNIST images")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--digit", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--model-dir", type=str, default="./output")
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()
    main(args)
