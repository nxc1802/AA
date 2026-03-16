import sys
import os
import torch
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.whisper_model import WhisperModelWrapper
from src.utils.metrics import calculate_wer
from src.data.dataset import LibriSpeechLoader

def main():
    source_audio_dir = "output_phase3/audio"
    output_dir = "output_phase3/transfer"
    os.makedirs(output_dir, exist_ok=True)
    
    # We test on Whisper Base and Whisper Small
    target_models = ["openai/whisper-base", "openai/whisper-small"]
    
    print("[*] Starting Transferability Evaluation...")
    dataset = LibriSpeechLoader(split="validation", num_samples=75)
    
    results = []
    
    for model_id in target_models:
        print(f"[*] Testing transfer to: {model_id}")
        wrapper = WhisperModelWrapper(model_id=model_id)
        
        success_count = 0
        total_count = 0
        
        for i in tqdm(range(75), desc=f"Transfer to {model_id.split('/')[-1]}"):
            audio_path = os.path.join(source_audio_dir, f"p3_sample_{i}_untargeted.wav")
            if not os.path.exists(audio_path):
                continue
                
            orig_audio, ground_truth = dataset.get_audio(i)
            
            # 1. Original transcription on target model
            orig_transcript = wrapper.transcribe(orig_audio)
            wer_orig = calculate_wer(ground_truth, orig_transcript)
            
            # 2. Adversarial transcription on target model
            import soundfile as sf
            adv_audio, _ = sf.read(audio_path)
            adv_transcript = wrapper.transcribe(adv_audio)
            wer_adv = calculate_wer(ground_truth, adv_transcript)
            
            success = wer_adv > wer_orig
            if success:
                success_count += 1
            total_count += 1
            
            results.append({
                "sample_id": i,
                "target_model": model_id,
                "wer_orig": wer_orig,
                "wer_adv": wer_adv,
                "success": success
            })
            
        print(f"Done {model_id}: TSR = {(success_count/total_count)*100:.2f}%")
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "transfer_results.csv"), index=False)
    
    summary = df.groupby("target_model")["success"].mean() * 100
    print("\nTRANSFERABILITY SUMMARY (TSR):")
    print(summary)
    
    summary.to_json(os.path.join(output_dir, "transfer_summary.json"))

if __name__ == "__main__":
    main()
