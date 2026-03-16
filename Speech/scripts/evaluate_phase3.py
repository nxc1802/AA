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
                               calculate_audio_quality)

def run_evaluation(model_name, dataset, target_text=None, is_targeted=False):
    print(f"[*] Benchmarking {model_name} (Targeted: {is_targeted})")
    model_wrapper = WhisperModelWrapper(model_id=model_name)
    
    attack = SubspaceHessianAttack(
        model_wrapper, 
        patch_percent=0.1, 
        step_strategy="decay", 
        num_iter=10,
        is_targeted=is_targeted
    )
    
    results = []
    
    for i in tqdm(range(len(dataset)), desc=f"Eval {model_name}"):
        audio_array, ground_truth = dataset.get_audio(i)
        
        # Reset counters
        model_wrapper.forward_count = 0
        model_wrapper.backward_count = 0
        
        start_time = time.time()
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        adv_audio_array = attack.attack(audio_tensor, ground_truth, target_text=target_text)
        duration = time.time() - start_time
        
        # Save audio for transferability testing
        audio_dir = "output_phase3/audio"
        os.makedirs(audio_dir, exist_ok=True)
        suffix = "targeted" if is_targeted else "untargeted"
        sf.write(os.path.join(audio_dir, f"p3_sample_{i}_{suffix}.wav"), adv_audio_array, 16000)
        
        # Evaluate on the SAME model (attacker model)
        adv_transcript = model_wrapper.transcribe(adv_audio_array)
        
        # Success Logic
        if is_targeted:
            # Targeted success: transcript matches target_text
            success = calculate_wer(target_text, adv_transcript) < 0.5 # Meaningfully similar
        else:
            # Untargeted success: WER increases
            wer_orig = calculate_wer(ground_truth, model_wrapper.transcribe(audio_array))
            wer_adv = calculate_wer(ground_truth, adv_transcript)
            success = wer_adv > wer_orig
            
        results.append({
            "id": i,
            "duration": duration,
            "fw_passes": model_wrapper.forward_count,
            "bw_passes": model_wrapper.backward_count,
            "success": success,
            "transcript": adv_transcript
        })
        
    return results, model_wrapper

def main():
    dataset = LibriSpeechLoader(split="validation", num_samples=75)
    
    # 1. Untargeted Efficiency Benchmark
    results_untargeted, wrapper = run_evaluation("openai/whisper-tiny", dataset, is_targeted=False)
    df_u = pd.DataFrame(results_untargeted)
    
    # 2. Targeted Attack Benchmark
    TARGET_CMD = "OPEN THE DOOR"
    results_targeted, _ = run_evaluation("openai/whisper-tiny", dataset, target_text=TARGET_CMD, is_targeted=True)
    df_t = pd.DataFrame(results_targeted)
    
    # Summary
    summary = {
        "untargeted_asr": float(df_u["success"].mean() * 100),
        "targeted_asr": float(df_t["success"].mean() * 100),
        "avg_fw_passes": float(df_u["fw_passes"].mean()),
        "avg_bw_passes": float(df_u["bw_passes"].mean()),
        "avg_duration": float(df_u["duration"].mean())
    }
    
    print("\n" + "="*30)
    print("PHASE 3: EVALUATION SUMMARY")
    print(f"Untargeted ASR: {summary['untargeted_asr']:.2f}%")
    print(f"Targeted ASR ('{TARGET_CMD}'): {summary['targeted_asr']:.2f}%")
    print(f"Efficiency: {summary['avg_fw_passes']} Forward, {summary['avg_bw_passes']} Backward passes")
    print("="*30)
    
    os.makedirs("output_phase3", exist_ok=True)
    with open("output_phase3/phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()
