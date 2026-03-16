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
from src.attacks.baselines import pgd_attack, fgsm_attack
from src.attacks.hessian_patch import SubspaceHessianAttack
from src.utils.metrics import (calculate_wer, calculate_cer, calculate_snr, 
                               calculate_lp_norms, calculate_audio_quality)

def evaluate_attack(model_wrapper, attack_name, attack_fn, dataset, output_dir):
    print(f"\n[*] Starting Benchmark: {attack_name}")
    results = []
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    for i in tqdm(range(len(dataset)), desc=f"Evaluating {attack_name}"):
        audio_array, ground_truth = dataset.get_audio(i)
        
        # Reset counters
        model_wrapper.forward_count = 0
        model_wrapper.backward_count = 0
        
        start_time = time.time()
        
        # Execute Attack
        if hasattr(attack_fn, 'attack'):
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
            adv_audio = attack_fn.attack(audio_tensor, ground_truth)
        else:
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
            adv_audio = attack_fn(model_wrapper, audio_tensor, ground_truth)
            
        duration = time.time() - start_time
        
        # Transcription
        adv_transcript = model_wrapper.transcribe(adv_audio)
        wer_adv = calculate_wer(ground_truth, adv_transcript)
        cer_adv = calculate_cer(ground_truth, adv_transcript)
        
        # Audio Quality
        pesq_val, stoi_val = calculate_audio_quality(audio_array, adv_audio)
        snr_val = calculate_snr(audio_array, adv_audio)
        
        # Success logic (Untargeted)
        success = wer_adv > 0.1 # Meaningful error
        
        results.append({
            "id": i,
            "wer": wer_adv,
            "cer": cer_adv,
            "pesq": pesq_val,
            "stoi": stoi_val,
            "snr": snr_val,
            "duration": duration,
            "fw_passes": model_wrapper.forward_count,
            "bw_passes": model_wrapper.backward_count,
            "success": success
        })
        
        # Save a few audio samples for audit
        if i < 3:
            sf.write(os.path.join(audio_dir, f"{attack_name.lower()}_sample_{i}.wav"), adv_audio, 16000)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"{attack_name.lower()}_detailed.csv"), index=False)
    
    summary = {
        "avg_wer": float(df["wer"].mean()),
        "avg_pesq": float(df["pesq"].mean()),
        "avg_stoi": float(df["stoi"].mean()),
        "avg_duration": float(df["duration"].mean()),
        "avg_fw": float(df["fw_passes"].mean()),
        "avg_bw": float(df["bw_passes"].mean()),
        "asr": float(df["success"].mean() * 100)
    }
    return summary

def main():
    output_base_dir = "output_final_report"
    os.makedirs(output_base_dir, exist_ok=True)
    
    dataset = LibriSpeechLoader(split="validation", num_samples=75)
    model_wrapper = WhisperModelWrapper(model_id="openai/whisper-tiny")
    
    # 1. Hessian-Patch (Subspace Contender - 10% patch)
    hessian_attack_sub = SubspaceHessianAttack(model_wrapper, patch_percent=0.1, num_iter=10, epsilon=0.005)
    summary_hessian_sub = evaluate_attack(model_wrapper, "Hessian-Subspace", hessian_attack_sub, dataset, output_base_dir)
    
    # 2. Hessian-Patch (Global Contender - 100% patch for FAIR comparison)
    hessian_attack_global = SubspaceHessianAttack(model_wrapper, patch_percent=1.0, num_iter=10, epsilon=0.005)
    summary_hessian_global = evaluate_attack(model_wrapper, "Hessian-Global", hessian_attack_global, dataset, output_base_dir)
    
    # 3. FGSM (Fast Baseline)
    def fgsm_wrapper(m, a, t): return fgsm_attack(m, a, t, epsilon=0.005)
    summary_fgsm = evaluate_attack(model_wrapper, "FGSM", fgsm_wrapper, dataset, output_base_dir)
    
    # 4. PGD-200 (The accurate but slow baseline)
    def pgd_wrapper(m, a, t): return pgd_attack(m, a, t, epsilon=0.005, num_iter=200)
    summary_pgd = evaluate_attack(model_wrapper, "PGD-200", pgd_wrapper, dataset, output_base_dir)
    
    final_summary = {
        "Hessian-Subspace": summary_hessian_sub,
        "Hessian-Global": summary_hessian_global,
        "FGSM": summary_fgsm,
        "PGD-200": summary_pgd
    }
    
    with open(os.path.join(output_base_dir, "final_comparison.json"), "w") as f:
        json.dump(final_summary, f, indent=4)
        
    print("\n" + "="*50)
    print("FINAL PIPELINE COMPLETE")
    print("Results saved to 'output_final_report'")
    print("="*50)

if __name__ == "__main__":
    main()
