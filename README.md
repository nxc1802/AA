# AA — Gradient-Guided Sparse Adversarial Perturbations

> **Paper**: *"Towards Sparse Adversarial Perturbations: A Gradient-Guided Approach for Efficient Image Attacks"*
> **Authors**: Xuan-Cuong Nguyen, Nhat-Quang Truong

Research codebase implementing and evaluating **sparse adversarial attacks** that selectively perturb only the most gradient-important pixels, achieving competitive Attack Success Rates while preserving image quality significantly better than dense attacks.

---

## Overview

Traditional adversarial attacks (FGSM, PGD, BIM) modify nearly every pixel in an image. This project demonstrates that **adversarial vulnerability is spatially concentrated** — high attack success rates can be achieved by modifying as few as 10% of pixels, with near-perfect perceptual quality (SSIM ≈ 0.995, PSNR > 42 dB).

**Key Contribution: Top-K Sparse PGD (`topk_pgd_attack`)**
- At each iteration, computes gradient magnitude as pixel importance score
- Applies perturbation only to the top-k% most important pixels
- Supports **dynamic masking**: k decreases over iterations for better convergence
- Also includes **GG-SAT**: a robust training method that defends against sparse attacks

---

## Project Structure

```
AA/
├── src/                        # Core library
│   ├── attacks/                # Attack implementations
│   │   ├── fgsm.py             # Fast Gradient Sign Method
│   │   ├── bim.py              # Basic Iterative Method
│   │   ├── pgd.py              # Projected Gradient Descent
│   │   └── topk_pgd.py         # ⭐ Top-K Sparse PGD (main contribution)
│   ├── defenses/               # Defense implementations
│   │   ├── base.py             # Abstract base class
│   │   └── preprocessing.py    # 6 defense strategies
│   ├── datasets/               # Dataset loaders
│   │   ├── loader.py           # CIFAR-10/100, Tiny-ImageNet, ImageNet
│   │   └── tiny_imagenet_setup.py
│   ├── models/                 # Model loaders
│   │   └── loader.py           # ResNet-18/50, TRADES, GG-SAT
│   └── utils/                  # Metrics & visualization
│       ├── metrics.py          # PSNR, SSIM, ASR, L0/L2/Linf
│       └── visualization.py    # 3-panel perturbation visualizer
├── scripts/                    # Experiment pipeline
│   ├── setup_datasets.py       # Step 1: Download/setup datasets
│   ├── generate_attacks.py     # Step 2: Generate & save adversarial images
│   ├── run_defense_bench.py    # Step 3: Evaluate defenses on saved attacks
│   ├── run_final_bench.py      # Step 4: Full k-ratio sweep benchmark
│   ├── make_paper_assets.py    # Step 5: Generate plots & LaTeX tables
│   ├── train_sparse_robust.py  # GG-SAT: Train sparse-robust model
│   └── eval_sparse_robust.py   # GG-SAT: Evaluate trained model
├── docs/                       # Documentation
│   ├── AA_Paper.md             # Full research paper
│   ├── GG_SAT_CLI_Guide.md     # CLI usage guide
│   ├── Sparse_Robust_Training.md # GG-SAT training guide
│   └── Defense_Analysis.md     # In-depth defense method analysis
├── results/                    # Experiment outputs (auto-generated)
└── requirements.txt
```

---

## Installation

```bash
# Clone repo and set up virtual environment
git clone <repo_url>
cd AA
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

**Dependencies**: `torch`, `torchvision`, `numpy`, `matplotlib`, `tqdm`, `scipy`, `requests`, `robustbench`, `pandas`

> **Note**: `robustbench` is used to auto-download pre-trained robust models (TRADES, Wong2020Fast). Without it, the system falls back to locally stored checkpoints.

---

## Quick Start

### 1. Setup Datasets
```bash
# Fast mock setup (no download, for testing code):
python scripts/setup_datasets.py --dataset all --mock

# Real CIFAR-10 (downloaded automatically by PyTorch):
# CIFAR-10 is auto-downloaded on first run of any experiment.

# Real Tiny-ImageNet:
python scripts/setup_datasets.py --dataset tiny_imagenet
```

### 2. Run the Full Benchmark
```bash
# Quick test (2 batches × 32 images):
python scripts/run_final_bench.py --batches 2 --batch_size 32

# Full paper benchmark (1000 samples):
python scripts/run_final_bench.py --batches 125 --batch_size 8
```
Output: `results/final_results.csv`

### 3. Generate Paper Assets
```bash
python scripts/make_paper_assets.py
```
Output: `results/paper_assets/*.png` + `*.tex`

### 4. Run Defense Benchmark
```bash
# Generate attacks first (saved as .pt files):
python scripts/generate_attacks.py --dataset cifar10 --model resnet18 --batches 8 --batch_size 128

# Evaluate all 7 defenses:
python scripts/run_defense_bench.py --dataset cifar10
```

### 5. Train GG-SAT Robust Model
```bash
# Quick test (1 epoch):
python scripts/train_sparse_robust.py --epochs 1 --batch_size 64 --val_size 32

# Full training (100 epochs, ~6-12 hours on GPU):
python scripts/train_sparse_robust.py --epochs 100 --batch_size 128 --k_min 0.3 --k_max 0.7

# Evaluate trained model vs baselines:
python scripts/eval_sparse_robust.py --batches 8 --batch_size 128
```

---

## Key Results (CIFAR-10, 1000 samples)

### Standard Model
| Attack | ASR (%) | SSIM | PSNR (dB) | Pixels Modified |
|--------|---------|------|-----------|-----------------|
| PGD-10 | 100.0 | 0.967 | 32.96 | ~100% |
| **Sparse k=0.1** | **91.5** | **0.995** | **42.41** | **~39%** |
| **Sparse k=0.3** | **99.4** | **0.987** | **38.12** | **~77%** |

### Robust Model (Adversarial Training)
| Attack | ASR (%) | SSIM | PSNR (dB) |
|--------|---------|------|-----------|
| PGD-10 | 31.0 | 0.967 | 30.51 |
| Sparse k=0.1 | 14.9 | 0.997 | 41.15 |

---

## Attack API

```python
from src.attacks import fgsm_attack, bim_attack, pgd_attack, topk_pgd_attack

# Standard PGD
adv = pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10)

# Sparse Top-K PGD (main contribution)
adv = topk_pgd_attack(
    model, images, labels,
    eps=8/255,      # Linf budget
    alpha=2/255,    # Step size
    iters=10,       # Number of iterations
    k_ratio=0.3,    # Fraction of pixels to perturb (0.1 = 10%)
    dynamic=True,   # Dynamic k scheduling (k decreases over iterations)
)
```

## Defense API

```python
from src.defenses import (
    MedianSmoothingDefense,
    BitReductionsDefense,
    JPEGCompressionDefense,
    RandomNoiseDefense,
    RandomizedSmoothingModel,
    FeatureDenoisingWrapper,
)

# Preprocessing defense (wraps input)
defense = MedianSmoothingDefense(kernel_size=3)
clean_input = defense(adversarial_images)
output = model(clean_input)

# Certified defense (wraps model)
smoothed = RandomizedSmoothingModel(model, sigma=0.12, N=100)
output = smoothed(adversarial_images)
```

---

## References

1. Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* arXiv:1412.6572
2. Kurakin et al. (2016). *Adversarial Examples in the Physical World.* arXiv:1607.02533
3. Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks.* arXiv:1706.06083
4. Papernot et al. (2016). *The Limitations of Deep Learning in Adversarial Settings.* EuroS&P 2016
5. Su et al. (2019). *One Pixel Attack for Fooling Deep Neural Networks.* IEEE TEC 23(5)
6. Zhang et al. (2019). *Theoretically Principled Trade-off between Robustness and Accuracy.* ICML 2019
