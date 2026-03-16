# Implementation Plan: Phase 2 - Subspace Hessian-Patch Algorithm

This document outlines the technical design and architectural steps required to implement the core innovation of the AA project: The Subspace Hessian-Patch Algorithm. This phase targets the audio 1D waveform using localized, second-order optimization.

## 1. Goal Description
The objective is to create an adversarial attack that is **computationally cheaper** than PGD-200 but achieves **equal or greater destruction** (WER degradation) by localizing the attack to a small audio segment (patch) and using approximated second-order curvature (Hessian).

## 2. Proposed Component Architecture

The implementation will be modularized into new files within the `src/attacks` directory to separate localization logic from optimization logic.

### 2.1. `src/attacks/localization.py`
- **[NEW] `class PatchLocator`**:
  - Implements the **Prefix-Sum + 1D Sliding Window** algorithm in $O(N)$ time.
  - **Input:** $g_0$ (Global 1st-order gradient of the 1D waveform) and `patch_percent` (e.g., 5% to 10% of total audio length).
  - **Process:** Calculates the magnitude array $|g_0|$, computes prefix sums, and slides a window of size `int(len(audio) * patch_percent)` to find the segment with the maximum cumulative gradient magnitude.
  - **Output:** Start and end indices of the most vulnerable subspace, and the geometric mask $M$.

### 2.2. `src/attacks/hessian_patch.py`
- **[NEW] Step-size Strategies**:
  - Implement a flexible system for Frank-Wolfe step sizes via a `Config` variable:
    - `fixed`: Constant $\gamma$.
    - `decay`: Standard $2/(k+2)$.
    - `line_search`: Exact line search for optimal progress.
- **[NEW] `def forward_euler_vhp(...)`**:
  - Implements the Truncated Vector-Hessian Product (vHp) approximation.
  - Formula: $H \cdot v \approx \frac{\nabla_x L(x + hv) - \nabla_x L(x)}{h}$ where $h$ is a small scalar.
  - **Crucial Optimization**: The computation of the perturbed gradient $\nabla_x L(x + hv)$ is strictly masked by $M$, ensuring backpropagation only computes gradients for the localized patch, saving massive memory.

- **[NEW] `class SubspaceHessianAttack`**:
  - The main orchestrator connecting localization and optimization.
  - **Step 1:** Calculate global $g_0$.
  - **Step 2:** Call `PatchLocator` to get mask $M$.
  - **Step 3:** Initialize perturbation using $g_0$ mapped to the subspace.
  - **Step 4 (Loop):** Execute the **Frank-Wolfe (FW)** algorithm:
    - Use Forward Euler to xấp xỉ bậc 2 cục bộ (local 2nd-order approximation).
    - Solve the linear minimization subproblem (LMO) in the FW step (projection-free).
    - Update the local patch noise.
  - **Output:** The final adversarial 1D audio waveform.

### 2.3. `scripts/evaluate_phase2.py`
- **[NEW] Configuration**: Loads the 75 baseline samples.
- **Action**: Runs the `SubspaceHessianAttack` on each sample.
- **Reporting**: Logs WER, CER, SNR, PESQ.
- **Crucial Metric**: Introduces a computational tracking metric (e.g., measuring forward/backward pass counts or exact execution elapsed time) to definitively prove that Hessian-Patch achieves the >90% WER of PGD-200 in a fraction of the time.

## 3. Verification Plan
- **Unit Testing**: Create a dummy sine wave and test if `PatchLocator` successfully identifies manually injected high-gradient segments.
- **Integration Testing**: Verify that applying the Subspace mask $M$ during backward passes actually reduces VRAM usage and execution time compared to a full-audio backward pass.
- **Benchmarking**: Compare results directly with `evaluation_results_pgd200.log`. Success is defined as achieving >90% WER with at least a 3x speedup in optimization time per sample.

## 4. User Review Required
> [!IMPORTANT]  
> 1. **Patch Size Formulation**: Do you want the `patch_length` to be defined as a fixed absolute duration (e.g., 2.0 seconds) or a relative percentage of the total audio per sample (e.g., 10%)?
> 2. **Frank-Wolfe Step Size**: FW typically requires a diminishing step-size rule like $\gamma_k = \frac{2}{k+2}$. Do you want to stick with this standard, or use an exact line-search since our problem is highly non-convex?
