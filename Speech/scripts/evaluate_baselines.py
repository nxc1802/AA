import sys
import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import LibriSpeechLoader
from src.models.whisper_model import WhisperModelWrapper
from src.attacks.baselines import pgd_attack
from src.utils.metrics import (calculate_wer, calculate_cer, calculate_snr, 
                               calculate_lp_norms, calculate_audio_quality)
from src.utils.visualization import plot_attack_comparison

def main():
    # Setup directories
    output_dir = "output_experiment"
    audio_dir = os.path.join(output_dir, "audio")
    viz_dir = os.path.join(output_dir, "visualization")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("Loading Dataset (75 samples)...")
    dataset = LibriSpeechLoader(split="validation", num_samples=75)
    
    print("Loading Whisper Tiny Model...")
    model_wrapper = WhisperModelWrapper(model_id="openai/whisper-tiny")
    
    epsilon_L_inf = 0.005
    iterations = 200
    
    results = []
    
    for i in tqdm(range(len(dataset)), desc="Evaluating Samples"):
        audio_array, true_text = dataset.get_audio(i)
        
        # 1. Original Transcription
        original_transcription = model_wrapper.transcribe(audio_array)
        wer_orig = calculate_wer(true_text, original_transcription)
        cer_orig = calculate_cer(true_text, original_transcription)
        
        # 2. PGD-200 Attack on 1D Waveform
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        adv_audio_array = pgd_attack(model_wrapper, audio_tensor, true_text, epsilon=epsilon_L_inf, num_iter=iterations)
        
        # 3. Adversarial Transcription
        adv_transcription = model_wrapper.transcribe(adv_audio_array)
        wer_adv = calculate_wer(true_text, adv_transcription)
        cer_adv = calculate_cer(true_text, adv_transcription)
        
        # 4. Calculate Advanced Metrics
        snr_val = calculate_snr(audio_array, adv_audio_array)
        l2_norm, linf_norm = calculate_lp_norms(audio_array, adv_audio_array)
        pesq_val, stoi_val = calculate_audio_quality(audio_array, adv_audio_array)
        
        # Success check (Untargeted ASR: WER increased or transcript changed meaningfully)
        is_success = wer_adv > wer_orig
        
        sample_result = {
            "id": i,
            "ground_truth": true_text,
            "orig_transcript": original_transcription,
            "adv_transcript": adv_transcription,
            "wer_orig": wer_orig,
            "wer_adv": wer_adv,
            "cer_orig": cer_orig,
            "cer_adv": cer_adv,
            "snr": snr_val,
            "l2_norm": l2_norm,
            "linf_norm": linf_norm,
            "pesq": pesq_val,
            "stoi": stoi_val,
            "success": is_success
        }
        results.append(sample_result)
        
        # 5. Save Outputs
        # Save audio
        sf.write(os.path.join(audio_dir, f"sample_{i}_orig.wav"), audio_array, 16000)
        sf.write(os.path.join(audio_dir, f"sample_{i}_adv.wav"), adv_audio_array, 16000)
        
        # Save visualization (only for first 5 to save time/space)
        if i < 5:
            plot_attack_comparison(
                audio_array, adv_audio_array, 
                f"Sample {i} (WER: {wer_orig:.2f} -> {wer_adv:.2f})",
                os.path.join(viz_dir, f"sample_{i}_comparison.png")
            )

    # 6. Aggregate Results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results_detailed.csv"), index=False)
    
    summary = {
        "avg_wer_orig": float(df["wer_orig"].mean()),
        "avg_wer_adv": float(df["wer_adv"].mean()),
        "avg_cer_orig": float(df["cer_orig"].mean()),
        "avg_cer_adv": float(df["cer_adv"].mean()),
        "avg_snr": float(df["snr"].mean()),
        "avg_pesq": float(df["pesq"].mean()),
        "avg_stoi": float(df["stoi"].mean()),
        "asr": float(df["success"].mean() * 100), # Adversarial Success Rate
        "total_samples": int(len(df))
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "="*30)
    print("EXPERIMENT COMPLETE")
    print(f"ASR (Adversarial Success Rate): {summary['asr']:.2f}%")
    print(f"Avg WER: {summary['avg_wer_orig']:.2f} -> {summary['avg_wer_adv']:.2f}")
    print(f"Avg PESQ: {summary['avg_pesq']:.2f}")
    print(f"Detailed results saved in: {output_dir}")
    print("="*30)

if __name__ == "__main__":
    main()
