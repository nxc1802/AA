# Task: Analyzing Research Project AA

- [x] Synthesize research objectives and methodology
- [x] Analyze technical implementation details (pipeline)
- [x] Create comprehensive research analysis report
- [x] Final summary and notification to user

# Task: Implement Phase 1 Baselines
- [x] Create Implementation Plan
- [x] Set up project structure and dependencies
- [x] Implement Data Loader (LibriSpeech)
- [x] Implement Model Wrapper (Whisper Tiny)
- [x] Implement Baseline Attacks (FGSM, PGD)
- [x] Create Evaluation Script

# Task: Run Baseline Experiment
- [x] Create virtual environment (venv)
- [x] Install dependencies in venv
- [x] Optimize Data Loader for local caching and subset
- [x] Run baseline evaluation script

# Task: Advanced Reporting and Visualization
- [x] Update dependencies (PESQ, STOI, torchmetrics)
- [x] Implement advanced metrics in `src/utils/metrics.py` (CER, PESQ, STOI, Lp norms)
- [x] Update `WhisperModelWrapper` for explicit device logging
- [x] Update `evaluate_baselines.py` for comprehensive reporting and recording
- [x] Implement visualization script for waveform/spectrogram comparison
- [x] Update `run_experiment.sh` for full reporting

# Task: Phase 2 - Subspace Hessian-Patch Algorithm
- [x] Design Technical Implementation Plan for Phase 2
- [x] Implement `PatchLocator` (Prefix-Sum + 1D Sliding Window) in `src/attacks/localization.py`
- [x] Implement Vector-Hessian Product (vHp) approximation via Forward Euler
- [x] Implement Frank-Wolfe optimization loop in `src/attacks/hessian_patch.py`
- [x] Create `scripts/evaluate_phase2.py`
- [x] Benchmark Hessian-Patch against PGD-200 (WER vs Compute Time)

# Task: Phase 3 - Evaluation, Targeted Attacks & Transferability
- [x] Design Technical Implementation Plan for Phase 3
- [x] Implement Targeted Attack support in `SubspaceHessianAttack`
- [ ] Implement Multi-Step Scheduler (REDO/EOS logic if applicable)
- [x] Conduct Full-Scale Benchmark (75 samples) across model sizes (Tiny, Base, Small)
- [x] Measure Efficiency (Compute Cost vs WER)
- [x] Generate Comparative Report and Final Phase 3 Walkthrough

# Task: Final Comprehensive Evaluation (75 samples)
- [x] Run `scripts/final_unified_eval.py` on full dataset
- [x] Aggregate results into Final Report Table
- [x] Final Project Completion Notification (Speech)

# Phase 4 & 5: Image Expansion & Advanced Refinements
- [x] Reorganize project into `Speech` and `Image` folders
- [x] Design technical plan for Image Hessian-Patch (2D)
- [x] Setup Datasets and Models
- [x] Implement `SubspaceHessianAttack2D` (vHp for Image Tensors)
- [x] Perceptual Visualization (Original vs. Adv vs. Noise)
- [x] Intelligent Localization (PatchLocator2D)
- [x] Final Cross-Domain Comparative Report

# Phase 6: Area-ASR Trade-off Analysis (Efficiency Play)
- [x] Implement `Image/scripts/area_tradeoff_benchmark.py`
- [x] Sweep Patch Area (10%, 25%, 50%, 100%) with fixed $N=10$
- [x] Prove "Decision Bottleneck" via Saliency-Patch
- [x] Final authoritative Scientific Report update

# Phase 7: Loss Landscape & Complexity Analysis
- [x] Implement `Image/scripts/complexity_benchmark.py`
- [x] Test Global Hessian vs. PGD on ResNet-20, 32, 44, 56 (Fixed $N=5$)
- [x] Prove "Curvature Advantage" in deeper architectures
- [x] Update Report with Complexity Proof
# Phase 8: Perceptual Distortion & Efficiency Analysis
- [x] Implement `Image/scripts/perceptual_benchmark.py`
- [x] Measure SSIM, L2 Norm, and L-inf on ResNet-152
- [x] Quantify the "Visual Tax" of Global PGD
- [x] Final Report Update with Perceptual Metrics

# Phase 9: Final Comprehensive Modality-Scaling Benchmark
- [x] Implement `Image/scripts/final_unified_benchmark.py`
- [x] Run benchmark (100 samples, all ResNets, both datasets)
- [x] Generate authoritative Summary Table
- [x] Final Project Handover and Documentation

# Phase 10: Surgical Pixel Optimization (Beyond Patches)
- [x] Implement `PixelLocator2D` (Saliency Ranking)
- [x] Implement `SurgicalHessianAttack2D` (Sparse-Saliency Support)
- [x] Benchmark: Patch vs. Surgical Pixel (ASR per unit area)
- [x] Final Report Update (Efficiency Record)

# Phase 11: Grand Unified Multi-Modal Benchmark
- [x] Implement `Speech/src/attacks/surgical_hessian.py` (Audio Saliency Support)
- [x] Implement `final_grand_benchmark.py` (Unified Cross-Domain Runner)
- [x] Run benchmark (100 samples/model, all datasets)
- [x] Generate Master Comparative Table
- [x] Project Conclusion and Final Scientific Delivery
