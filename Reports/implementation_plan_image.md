# Implementation Plan: Hessian-Patch for Image Domain (Phase 4)

This phase extends the **Subspace Hessian-Patch** algorithm from 1D audio waveforms to 2D image data. We aim to prove that localized 2nd-order optimization is equally effective and efficient for image classification.

## 1. Multi-Dataset Support
We will implement data loaders for:
- **CIFAR-10**: 32x32 resolution, 10 classes. Standard benchmark for fundamental algorithm verification.
- **Tiny ImageNet**: 64x64 resolution, 200 classes. A more complex dataset for testing scalability.

## 2. Multi-Model Architecture
We will use the **ResNet** family to evaluate cross-architecture robustness:
- **Source Model**: `resnet18` (Lightweight, used for patch generation).
- **Transfer Targets**: `resnet34`, `resnet50` (Deeper models, used to evaluate **Transfer Success Rate (TSR)**).

## 3. Proposed Changes

### Image Module Structure
- **[NEW] `Image/src/data/loader.py`**:
    - `get_cifar10_loader()`: Load CIFAR-10 validation set.
    - `get_tiny_imagenet_loader()`: Load Tiny ImageNet validation set.
- **[NEW] `Image/src/models/vision_wrapper.py`**:
    - Generic `VisionModelWrapper` for ResNet models with `forward_count` and `backward_count`.
- **[NEW] `Image/src/attacks/hessian_patch_2d.py`**:
    - **`forward_euler_vhp_2d`**: Higher-order curvature calculation for 4D Image Tensors (B, C, H, W).
    - **`SubspaceHessianAttack2D`**: Localized optimization using a spatial mask (e.g., center 10% patch).

### Evaluation & Benchmarking
- **[NEW] `Image/scripts/evaluate_image_baselines.py`**:
    - Comparative run: **Hessian-Patch (Subspace)** vs **PGD (Global)** vs **FGSM (Global)**.
- **[NEW] `Image/scripts/benchmark_vision_transfer.py`**:
    - Generate patches on `resnet18` $\to$ Test on `resnet34` and `resnet50`.

## 4. Metrics & Reporting
- **Performance**: Accuracy, ASR (Adversarial Success Rate).
- **Image Quality**: **SSIM** (Structural Similarity), **LPIPS** (Learned Perceptual Image Patch Similarity).
- **Efficiency**: Forward/Backward passes per sample.

## Phase 5: Advanced Image Refinements
Enhancing the Image expansion with perceptual stealth analysis and advanced optimization control.

### 1. Perceptual Visualization
- **[NEW] `Image/scripts/visualize_attack.py`**: A dedicated utility to generate side-by-side comparisons of (Original | Adversarial | Perturbation).
- Inclusion of **SSIM** and **MSE** metrics in the visualization titles.

### 2. Intelligent Localization (PatchLocator2D)
- **[NEW] `Image/src/attacks/localization_2d.py`**:
    - Implementation of a sliding-window gradient accumulator.
    - Finds the $H \times W$ region with the highest cumulative gradient magnitude for maximum disruption.

### 3. Targeted Attack Support
- Update `SubspaceHessianAttack2D`:
    - Support for `is_targeted` flag.
    - Implementation of loss minimization towards a target class using 2nd-order curvature.

### 4. Tiny ImageNet Infrastructure
- Robust downloader/setup script for Tiny ImageNet.
- Benchmarking on 64x64 resolution.

## Phase 6: Area-ASR Trade-off Analysis
This phase aims to prove that localized patches can achieve high ASR by focusing on "Decision Bottlenecks," thereby justifying the use of Subspace constraints as a cost-reduction strategy for expensive attacks (like Hessian or complex PGD ensembles).

### 1. Parametric Area Sweep
- **Script**: `Image/scripts/area_tradeoff_benchmark.py`.
- **Logic**: For a fixed $N=10$, evaluate ASR for:
    - 10% (Subspace/Stealth)
    - 25% (Localized)
    - 50% (Major Saliency)
    - 100% (Global/Baseline)

### 2. Efficiency Metrics
- **Impact Density**: ASR divided by Area. A higher value proves the "Efficiency" of the Patch mechanism.
- **Cost Equivalence**: Demonstrating that a 50% smart-patch (using `PatchLocator2D`) can outperform a random 100% noise field at low iterations.
## Phase 10: Surgical Pixel Optimization (Beyond Patches)
This phase evolves the geometric "Patch" constraint into a functional "Pixel Saliency" constraint.

### 1. Saliency Ranking Strategy
- **Mechanism**: Instead of finding a rectangular box, we rank all pixels $(i, j)$ by their gradient magnitude $||\nabla_x L||$.
- **Constraint**: We select only the top $K$% of pixels (e.g., $K=10$) to form a sparse adversarial mask.
- **Advantage**: Eliminates "wasteful" noise on unimportant pixels within a patch and captures high-impact pixels outside the patch boundaries.

### 2. Implementation
- **PixelLocator2D**: Computes a binary mask by thresholding based on top-$K$ percentile of the absolute gradient sum across channels.
- **SurgicalHessianAttack2D**: Integrates the sparse mask into the Frank-Wolfe optimization loop.

### 3. Comparison Metrics
- **Impact Density**: ASR divided by the number of modified pixels.
- **L2 Efficiency**: Loss increase per unit of L2 distortion.
