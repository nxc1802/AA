# Implementation Plan: Phase 3 - Attack Refinement & Extensive Evaluation

This phase expands the **Subspace Hessian-Patch** algorithm to handle complex attack scenarios (Targeted Attacks) and rigorously validates its efficiency and transferability across different Whisper model scales.

## 1. Goal Description
The objective of Phase 3 is to prove that the Hessian-Patch attack is not only faster than PGD but also more versatile (Targeted) and capable of "transferring" its effectiveness to larger, unobserved models.

## 2. Proposed Changes

### 2.1. Refinement of Loss Functions (`src/attacks/hessian_patch.py`)
- **[MODIFY] `SubspaceHessianAttack`**:
  - Add support for **Targeted Attack Mode**: Instead of maximizing Cross-Entropy loss from the ground truth, minimize Cross-Entropy loss towards a `target_transcript` (e.g., "OK Google, open evil.com").
  - **REDO/EOS Integration (Optional)**: If targeting specific decoding failures, implement specific loss penalties for the "End of Sentence" token to prevent premature termination or force infinite loops (REDO).

### 2.2. Transferability & Scale Testing (`scripts/evaluate_phase3.py`)
- **[NEW] Multi-Model Benchmarking**: 
  - Execute the same patch generated on **Whisper Tiny** and test it against **Whisper Base** and **Whisper Small**.
  - **Metric**: Transfer Success Rate (TSR) — How often a patch designed for Tiny still breaks Small?

### 2.3. Efficiency & Cost Analysis
- **[MODIFY] Metrics Module (`src/utils/metrics.py`)**:
  - Add logic to count `forward()` and `backward()` calls during optimization.
  - **Goal**: Formally verify the $N+1$ pass claim from the project pipeline vs the $10$ to $200$ passes used by PGD.

## 3. Verification Plan

### Automated Benchmarks
- **Test 3.1 (Targeted)**: Attempt to force 20 samples to transcribe a specific command. Measure **Target Success Rate**.
- **Test 3.2 (Efficiency)**: Compare `Compute-to-Noise` ratio. How much WER change do we get per unit of GPU time?
- **Test 3.3 (Scale)**: Evaluate on the full 75 LibriSpeech samples using the optimized Phase 2 code.

## 4. Final Goal
Produce a **Comparative Performance Matrix** showing:
1. **PGD-200** (Gold standard for destruction, high cost)
2. **PGD-10** (Baseline for speed, moderate destruction)
3. **Hessian-Patch-10** (Target for Phase 3: Low cost, high destruction, high stealth).

## 5. User Review Required
> [!IMPORTANT]
> 1. **Target String**: For the Targeted Attack tests, do you have a specific "malicious" string you want all audio to transcribe into? (e.g., "Mở cửa phòng" or "OK Google").
> 2. **Transferability Models**: Do you want to test transferability up to `whisper-large-v3`, or is `whisper-base/small` sufficient for this phase?
