"""
Evaluate quality of generated MNIST data.

Trains a CNN classifier on synthetic data from the flow matching model,
then evaluates on real and synthetic datasets. Optionally trains a baseline
classifier on real data for comparison. Outputs confusion matrices,
correlation matrices, and accuracy comparison plots.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from model import UNet, MNISTClassifier
from sample import sample


# ---------- Synthetic data generation ----------

def generate_synthetic_dataset(gen_model, device, samples_per_class=6000,
                               n_steps=100, cfg_scale=2.0, batch_size=500):
    """Generate a full labeled synthetic dataset."""
    gen_model.eval()
    all_images = []
    all_labels = []

    for digit in range(10):
        print(f"  Generating digit {digit}...")
        images_for_digit = []
        remaining = samples_per_class

        while remaining > 0:
            n = min(batch_size, remaining)
            imgs = sample(gen_model, device, n_samples=n, n_steps=n_steps,
                          class_label=digit, cfg_scale=cfg_scale)
            imgs = imgs * 2 - 1  # [0,1] -> [-1,1]
            images_for_digit.append(imgs.cpu())
            remaining -= n

        all_images.append(torch.cat(images_for_digit, dim=0)[:samples_per_class])
        all_labels.append(torch.full((samples_per_class,), digit, dtype=torch.long))

    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    perm = torch.randperm(len(images))
    return images[perm], labels[perm]


# ---------- Classifier training & evaluation ----------

def train_classifier(model, loader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        print(f"  Epoch {epoch}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {100*correct/total:.1f}%")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    acc = 100 * correct / total
    return acc, torch.cat(all_preds), torch.cat(all_labels)


# ---------- Plotting ----------

def compute_confusion_matrix(preds, labels, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds.numpy(), labels.numpy()):
        cm[l, p] += 1
    return cm


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(cm, title, path):
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    corr = np.corrcoef(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f'{corr[i, j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(corr[i, j]) > 0.5 else 'black')

    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Digit', fontsize=12)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(results, path):
    """Bar chart comparing accuracies."""
    names = list(results.keys())
    accs = [results[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
    bars = ax.bar(names, accs, color=colors[:len(names)], width=0.5)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Real Test Accuracy by Training Data Source', fontsize=14)
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------- Main ----------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load real datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    real_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    real_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    real_train_loader = DataLoader(real_train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    real_test_loader = DataLoader(real_test, batch_size=256, num_workers=4, pin_memory=True)

    results = {}

    # --- Synthetic-trained classifier ---
    print("\nLoading flow matching model...")
    gen_model = UNet(base_channels=args.channels).to(device)
    gen_model.load_state_dict(torch.load(
        os.path.join(args.model_dir, "model.pt"),
        map_location=device, weights_only=True))

    print(f"\nGenerating synthetic dataset ({args.samples_per_class} per class)...")
    syn_images, syn_labels = generate_synthetic_dataset(
        gen_model, device,
        samples_per_class=args.samples_per_class,
        n_steps=args.sample_steps,
        cfg_scale=args.cfg_scale)
    print(f"  Synthetic dataset: {len(syn_images)} images")

    syn_dataset = TensorDataset(syn_images, syn_labels)
    syn_train_loader = DataLoader(syn_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    syn_eval_loader = DataLoader(syn_dataset, batch_size=256, num_workers=4, pin_memory=True)

    print("\nTraining classifier on SYNTHETIC data...")
    syn_classifier = MNISTClassifier().to(device)
    train_classifier(syn_classifier, syn_train_loader, device, epochs=args.clf_epochs)

    print("\nEvaluating synthetic-trained classifier...")
    acc, preds, labels = evaluate(syn_classifier, real_train_loader, device)
    print(f"  Real Train Accuracy:  {acc:.2f}%")
    cm_syn_on_real_train = compute_confusion_matrix(preds, labels)

    acc, preds, labels = evaluate(syn_classifier, real_test_loader, device)
    results['Synthetic\nTraining'] = acc
    print(f"  Real Test Accuracy:   {acc:.2f}%")
    cm_syn_on_real_test = compute_confusion_matrix(preds, labels)

    acc, preds, labels = evaluate(syn_classifier, syn_eval_loader, device)
    print(f"  Synthetic Accuracy:   {acc:.2f}%")
    cm_syn_on_syn = compute_confusion_matrix(preds, labels)

    # --- Real-trained classifier ---
    print("\nTraining classifier on REAL data...")
    real_classifier = MNISTClassifier().to(device)
    train_classifier(real_classifier, real_train_loader, device, epochs=args.clf_epochs)

    print("\nEvaluating real-trained classifier...")
    acc, preds, labels = evaluate(real_classifier, real_test_loader, device)
    results['Real\nTraining'] = acc
    print(f"  Real Test Accuracy:   {acc:.2f}%")
    cm_real_on_real_test = compute_confusion_matrix(preds, labels)

    # --- Plots ---
    print("\nSaving plots...")
    plot_confusion_matrix(cm_syn_on_real_train, 'Synthetic-Trained — Real Train Data',
                          os.path.join(args.output_dir, 'cm_syn_on_real_train.png'))
    plot_confusion_matrix(cm_syn_on_real_test, 'Synthetic-Trained — Real Test Data',
                          os.path.join(args.output_dir, 'cm_syn_on_real_test.png'))
    plot_confusion_matrix(cm_syn_on_syn, 'Synthetic-Trained — Synthetic Data',
                          os.path.join(args.output_dir, 'cm_syn_on_syn.png'))
    plot_confusion_matrix(cm_real_on_real_test, 'Real-Trained — Real Test Data',
                          os.path.join(args.output_dir, 'cm_real_on_real_test.png'))

    plot_correlation_matrix(cm_syn_on_real_test, 'Class Correlation — Synthetic-Trained',
                            os.path.join(args.output_dir, 'corr_syn.png'))
    plot_correlation_matrix(cm_real_on_real_test, 'Class Correlation — Real-Trained',
                            os.path.join(args.output_dir, 'corr_real.png'))

    plot_comparison(results, os.path.join(args.output_dir, 'comparison.png'))

    # Summary
    gap = results['Real\nTraining'] - results['Synthetic\nTraining']
    print(f"\n{'='*40}")
    print(f"SUMMARY (Real Test Accuracy)")
    print(f"{'='*40}")
    print(f"  Trained on synthetic: {results['Synthetic'+chr(10)+'Training']:.2f}%")
    print(f"  Trained on real:      {results['Real'+chr(10)+'Training']:.2f}%")
    print(f"  Gap:                  {gap:.2f}%")
    print(f"{'='*40}")
    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic MNIST data quality")
    parser.add_argument("--model-dir", default="./output")
    parser.add_argument("--output-dir", default="./output/eval")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=6000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--clf-epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
