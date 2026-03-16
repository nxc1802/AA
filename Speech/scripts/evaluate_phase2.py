import sys
import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import json
import time
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import LibriSpeechLoader
from src.models.whisper_model import WhisperModelWrapper
from src.attacks.hessian_patch import SubspaceHessianAttack
from src.utils.metrics import (calculate_wer, calculate_cer, calculate_snr, 
                               calculate_lp_norms, calculate_audio_quality)
from src.utils.visualization import plot_attack_comparison

def main():
    # Configuration
    PATCH_PERCENT = 0.1 # 10% patch
    NUM_ITER = 10       # Phase 2 goal: 10 iterations to beat PGD-200
    STRATEGY = "decay"
    
    output_dir = "output_phase2"
    audio_dir = os.path.join(output_dir, "audio")
    viz_dir = os.path.join(output_dir, "visualization")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print(f"[*] Starting Phase 2 Evaluation: Hessian-Patch ({NUM_ITER} iterations, {PATCH_PERCENT*100}% patch)")
    dataset = LibriSpeechLoader(split="validation", num_samples=10) # Testing on first 10 for quick benchmark
    model_wrapper = WhisperModelWrapper(model_id="openai/whisper-tiny")
    
    attack = SubspaceHessianAttack(
        model_wrapper, 
        patch_percent=PATCH_PERCENT, 
        step_strategy=STRATEGY, 
        num_iter=NUM_ITER
    )
    
    results = []
    
    for i in tqdm(range(len(dataset)), desc="Processing Phase 2"):
        audio_array, true_text = dataset.get_audio(i)
        
        # 1. Original
        orig_transcript = model_wrapper.transcribe(audio_array)
        wer_orig = calculate_wer(true_text, orig_transcript)
        
        # 2. Attack with Timing
        start_time = time.time()
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        adv_audio_array = attack.attack(audio_tensor, true_text)
        attack_duration = time.time() - start_time
        
        # 3. Adversarial
        adv_transcript = model_wrapper.transcribe(adv_audio_array)
        wer_adv = calculate_wer(true_text, adv_transcript)
        
        # 4. Metrics
        pesq_v, stoi_v = calculate_audio_quality(audio_array, adv_audio_array)
        snr_v = calculate_snr(audio_array, adv_audio_array)
        
        results.append({
            "id": i,
            "wer_orig": wer_orig,
            "wer_adv": wer_adv,
            "duration": attack_duration,
            "pesq": pesq_v,
            "snr": snr_v,
            "success": wer_adv > wer_orig
        })
        
        # Save audio
        sf.write(os.path.join(audio_dir, f"p2_sample_{i}_adv.wav"), adv_audio_array, 16000)
        
        if i < 3:
            plot_attack_comparison(
                audio_array, adv_audio_array,
                f"Phase 2 - Hessian Patch Sample {i}",
                os.path.join(viz_dir, f"p2_sample_{i}.png")
            )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "phase2_detailed.csv"), index=False)
    
    print("\n" + "="*30)
    print("PHASE 2 BENCHMARK RESULTS")
    print(f"Avg WER: {df['wer_adv'].mean():.2f}")
    print(f"Avg Duration per sample: {df['duration'].mean():.2f}s")
    print(f"ASR: {df['success'].mean()*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()
