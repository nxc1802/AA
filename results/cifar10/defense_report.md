# Comprehensive Robustness Evaluation: CIFAR10 Defenses vs. Sparse Attacks

This report evaluates the effectiveness of standard preprocessing, certified, and feature-space defenses using **pre-saved decoupled adversarial images**.

## 1. Experimental Setup
- **Dataset**: cifar10
- **Methodology**: Decoupled attack generation and defense evaluation.
- **Total Samples Evaluated**: 100 samples.

## 2. Evaluation Results

### Results on Standard ResNet-18

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 11.0% | 11.0% | 9.09% | 11.0% | 0.0% | 11.0% | 9.09% | 11.0% | 11.0% |
| Median Filter (3x3) | 12.0% | 11.0% | 27.27% | 12.0% | 18.18% | 11.0% | 27.27% | 12.0% | 12.0% |
| Bit Reduction (3-bit) | 13.0% | 9.0% | 27.27% | 10.0% | 27.27% | 9.0% | 27.27% | 10.0% | 10.0% |
| JPEG Compression (Q75) | 10.0% | 13.0% | 0.0% | 10.0% | 9.09% | 13.0% | 0.0% | 10.0% | 10.0% |
| Random Noise (std=0.02) | 10.0% | 10.0% | 18.18% | 10.0% | 18.18% | 9.0% | 27.27% | 11.0% | 11.0% |
| Randomized Smoothing (std=0.12, N=100) | 10.0% | 9.0% | 45.45% | 10.0% | 45.45% | 10.0% | 45.45% | 10.0% | 10.0% |
| Feature Denoising (3x3 hooks) | 7.0% | 6.0% | 72.73% | 6.0% | 81.82% | 6.0% | 72.73% | 6.0% | 6.0% |


### Results on Robust ResNet-18 (AT)

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 89.0% | 90.0% | 0.0% | 89.0% | 0.0% | 90.0% | 0.0% | 89.0% | 89.0% |
| Median Filter (3x3) | 82.0% | 81.0% | 10.11% | 81.0% | 10.11% | 81.0% | 10.11% | 81.0% | 81.0% |
| Bit Reduction (3-bit) | 89.0% | 88.0% | 2.25% | 88.0% | 2.25% | 88.0% | 2.25% | 88.0% | 88.0% |
| JPEG Compression (Q75) | 87.0% | 86.0% | 3.37% | 87.0% | 3.37% | 86.0% | 3.37% | 87.0% | 87.0% |
| Random Noise (std=0.02) | 90.0% | 90.0% | 0.0% | 89.0% | 1.12% | 90.0% | 0.0% | 90.0% | 90.0% |
| Randomized Smoothing (std=0.12, N=100) | 70.0% | 70.0% | 22.47% | 70.0% | 22.47% | 71.0% | 21.35% | 70.0% | 70.0% |
| Feature Denoising (3x3 hooks) | 89.0% | 90.0% | 0.0% | 89.0% | 0.0% | 90.0% | 0.0% | 89.0% | 89.0% |


### Results on TRADES robust model

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 85.0% | 54.0% | 36.47% | 74.0% | 12.94% | 85.0% | 0.0% | 85.0% | 85.0% |
| Median Filter (3x3) | 78.0% | 60.0% | 29.41% | 71.0% | 16.47% | 79.0% | 7.06% | 77.0% | 77.0% |
| Bit Reduction (3-bit) | 85.0% | 53.0% | 37.65% | 74.0% | 12.94% | 87.0% | 0.0% | 86.0% | 86.0% |
| JPEG Compression (Q75) | 83.0% | 59.0% | 30.59% | 79.0% | 7.06% | 83.0% | 2.35% | 83.0% | 83.0% |
| Random Noise (std=0.02) | 85.0% | 55.0% | 35.29% | 74.0% | 12.94% | 86.0% | 0.0% | 85.0% | 85.0% |
| Randomized Smoothing (std=0.12, N=100) | 71.0% | 48.0% | 44.71% | 63.0% | 28.24% | 70.0% | 20.0% | 71.0% | 71.0% |
| Feature Denoising (3x3 hooks) | 85.0% | 54.0% | 36.47% | 74.0% | 12.94% | 85.0% | 0.0% | 85.0% | 85.0% |


### Results on GG-SAT ResNet-18 (GG-SAT)

| Defense | Clean Acc | PGD Direct Acc | PGD Direct ASR | Sparse Direct Acc | Sparse Direct ASR | PGD Transfer Acc | PGD Transfer ASR | Sparse Transfer Acc | Sparse Transfer ASR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Defense | 10.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% |
| Median Filter (3x3) | 10.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% |
| Bit Reduction (3-bit) | 10.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% |
| JPEG Compression (Q75) | 10.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% |
| Random Noise (std=0.02) | 10.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% |
| Randomized Smoothing (std=0.12, N=100) | 10.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% |
| Feature Denoising (3x3 hooks) | 10.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% |


