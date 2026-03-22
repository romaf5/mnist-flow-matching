# MNIST Flow Matching

MNIST digit generator using **Flow Matching** based on [An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070).

Flow matching learns a vector field that transports samples from Gaussian noise to the data distribution via straight-line (Conditional Optimal Transport) paths. Simpler than classic DDPM — no noise schedules, no reverse SDE, just regression.

## Results

A classifier trained **only on synthetic data** achieves **96.35%** on real MNIST test set, vs **99.24%** from real data training (2.89% gap).

## Usage

```bash
# Install
pip install -r requirements.txt

# Train generative model (20 epochs, ~2 min on RTX 3090)
python train.py --epochs 20

# Generate images
python sample.py --n-samples 64 --digit 7

# Evaluate: train classifier on synthetic, compare with real
python evaluate.py
```

## Architecture

- **Generator**: U-Net (~2M params) with sinusoidal time embeddings and adaptive normalization
- **Class conditioning**: Learned digit embeddings + classifier-free guidance (CFG)
- **Sampling**: Euler integration, 100 steps, CFG scale 2.0

## Files

| File | Description |
|------|-------------|
| `model.py` | U-Net and CNN classifier architectures |
| `train.py` | Flow matching training loop |
| `sample.py` | Image generation via Euler integration |
| `evaluate.py` | Synthetic vs real data evaluation with plots |
