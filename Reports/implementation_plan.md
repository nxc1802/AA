# Implementation Plan: Phase 1 Baselines

This plan outlines the implementation of Phase 1 of the AA project: setting up the baseline attacks (FGSM, PGD) on the Whisper Tiny model using the LibriSpeech dataset.

## Target Goals
- Set up the environment and dependencies (PyTorch, Transformers, Torchaudio, Datasets).
- Implement data loading and preprocessing for LibriSpeech (16kHz, 1D waveform).
- Integrate the OpenAI Whisper Tiny model.
- Implement baseline first-order adversarial attacks (FGSM and PGD-10).
- Create an evaluation script to measure Word Error Rate (WER) and Signal-to-Noise Ratio (SNR).

## Proposed Architecture

The code will be structured in a modular way within the `src` directory to allow for future expansion into Subspace Hessian algorithms.

### 1. `src/data/dataset.py`
- [NEW] Script to load and preprocess the LibriSpeech dataset.
- Functions to ensure audio is 16kHz and formatted correctly for Whisper.

### 2. `src/models/whisper_model.py`
- [NEW] Wrapper for loading the `openai/whisper-tiny` model from HuggingFace Transformers.
- Includes functions for processing input audio and generating transcriptions.

### 3. `src/attacks/baselines.py`
- [NEW] Implementation of standard first-order attacks.
- `FGSM`: Fast Gradient Sign Method.
- `PGD`: Projected Gradient Descent (configured for 10 iterations by default).

### 4. `src/utils/metrics.py`
- [NEW] Calculation of key metrics.
- Word Error Rate (WER).
- Signal-to-Noise Ratio (SNR).

### 5. `scripts/evaluate_baselines.py`
- [NEW] Main entry point for running the baseline attacks on the dataset and evaluating the results.

## Dependencies Needed (requirements.txt)
- `torch`, `torchaudio`
- `transformers`
- `datasets`
- `jiwer` (for WER calculation)
- `soundfile`

## User Review Required
> [!IMPORTANT]
> Please review this structure. Do you prefer using HuggingFace's `transformers` library for Whisper, or the official `openai-whisper` package? HuggingFace is generally easier for computing gradients w.r.t to the input audio for adversarial attacks.
