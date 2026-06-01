# Comprehensive Robustness Evaluation: TINY_IMAGENET Defenses vs. Sparse Attacks

This report evaluates the effectiveness of standard preprocessing, certified, and feature-space defenses using **pre-saved decoupled adversarial images**.

## 1. Experimental Setup
- **Dataset**: tiny_imagenet
- **Methodology**: Decoupled attack generation and defense evaluation.
- **Total Samples Evaluated**: 1 samples.

## 2. Evaluation Results

### Results on Standard ResNet-18

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Median Filter (3x3) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Bit Reduction (3-bit) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| JPEG Compression (Q75) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Random Noise (std=0.02) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Randomized Smoothing (std=0.12, N=100) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Feature Denoising (3x3 hooks) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |


### Results on Robust ResNet-18 (AT)

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Median Filter (3x3) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Bit Reduction (3-bit) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| JPEG Compression (Q75) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Random Noise (std=0.02) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Randomized Smoothing (std=0.12, N=100) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Feature Denoising (3x3 hooks) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |


### Results on TRADES robust model

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Median Filter (3x3) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Bit Reduction (3-bit) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| JPEG Compression (Q75) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Random Noise (std=0.02) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Randomized Smoothing (std=0.12, N=100) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Feature Denoising (3x3 hooks) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |


### Results on GG-SAT ResNet-18 (GG-SAT)

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Median Filter (3x3) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Bit Reduction (3-bit) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| JPEG Compression (Q75) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Random Noise (std=0.02) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Randomized Smoothing (std=0.12, N=100) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Feature Denoising (3x3 hooks) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |


