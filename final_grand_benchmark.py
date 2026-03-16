import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Add Root path for absolute package imports
sys.path.append(os.getcwd())

# Audio Imports
from Speech.src.data.dataset import LibriSpeechLoader # Fix import
from Speech.src.models.whisper_model import WhisperModelWrapper # Fix import
from Speech.src.attacks.baselines import fgsm_attack, pgd_attack
from Speech.src.attacks.hessian_patch import SubspaceHessianAttack
from Speech.src.attacks.surgical_hessian import SurgicalHessianAttack
from Speech.src.utils.metrics import calculate_audio_quality, calculate_wer

# Vision Imports
from Image.src.data.loader import get_cifar10_loader
from Image.src.data.tiny_loader import get_tiny_imagenet_loader
from Image.src.models.vision_wrapper import VisionModelWrapper
from Image.src.attacks.baselines_2d import pgd_attack_2d, fgsm_attack_2d
from Image.src.attacks.hessian_patch_2d import SubspaceHessianAttack2D
from Image.src.attacks.surgical_hessian_2d import SurgicalHessianAttack2D
from skimage.metrics import structural_similarity as ssim

def denorm_vision(tensor, is_cifar=True):
    if is_cifar:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(tensor.device)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def run_audio_benchmark(num_samples=100):
    print("\n[SPEECH] Running Grand Audio Benchmark...")
    wrapper = WhisperModelWrapper(model_id="openai/whisper-tiny")
    loader = LibriSpeechLoader(num_samples=num_samples) # Use the class directly
    epsilon = 0.005 
    num_iter = 10
    
    attacks = {
        "FGSM": lambda w, a, t: fgsm_attack(w, a, t, epsilon=epsilon),
        "PGD": lambda w, a, t: pgd_attack(w, a, t, epsilon=epsilon, num_iter=num_iter),
        "H-Patch": SubspaceHessianAttack(wrapper, patch_percent=0.1, num_iter=num_iter, epsilon=epsilon),
        "H-Surgical": SurgicalHessianAttack(wrapper, top_k_percent=0.1, num_iter=num_iter, epsilon=epsilon)
    }
    
    rows = []
    for name, attack in attacks.items():
        print(f"  > Method: {name}")
        asr_count = 0
        total_wer = []
        total_pesq = []
        total_l2 = []
        
        for i in tqdm(range(len(loader)), desc=f"Audio {name}"):
            audio_np, text = loader.get_audio(i)
            audio = torch.tensor(audio_np).unsqueeze(0).to(wrapper.device)
            
            # Attack
            if callable(attack) and not hasattr(attack, 'attack'):
                adv_audio = attack(wrapper, audio, text)
            else:
                adv_audio = attack.attack(audio, text)
            
            # Predict
            adv_text = wrapper.transcribe(adv_audio)
            wer = calculate_wer(text, adv_text)
            pesq, _ = calculate_audio_quality(audio_np, adv_audio)
            l2 = np.linalg.norm((audio_np - adv_audio).flatten())
            
            if wer > 0.1: asr_count += 1
            total_wer.append(wer)
            total_pesq.append(pesq)
            total_l2.append(l2)
            
        rows.append({
            "Modality": "Audio",
            "Model": "Whisper-Tiny",
            "Method": name,
            "ASR": (asr_count/len(loader))*100,
            "Quality (SSIM/PESQ)": np.mean(total_pesq),
            "L2 Energy": np.mean(total_l2),
            "WER/Other": np.mean(total_wer)
        })
    return rows

def run_vision_benchmark(model_name, loader, is_cifar=True, num_samples=100):
    print(f"\n[VISION] Benchmarking: {model_name}")
    wrapper = VisionModelWrapper(model_name=model_name)
    epsilon = 0.031 # fixed for vision
    num_iter = 10
    
    attacks = {
        "FGSM": lambda w, i, l: fgsm_attack_2d(w, i, l, epsilon=epsilon),
        "PGD": lambda w, i, l: pgd_attack_2d(w, i, l, epsilon=epsilon, num_iter=num_iter),
        "H-Patch": SubspaceHessianAttack2D(wrapper, patch_percent=0.1, num_iter=num_iter, epsilon=epsilon),
        "H-Surgical": SurgicalHessianAttack2D(wrapper, top_k_percent=0.1, num_iter=num_iter, epsilon=epsilon)
    }
    
    rows = []
    for name, attack in attacks.items():
        asr_count = 0
        ssims = []
        l2s = []
        
        count = 0
        for images, labels in tqdm(loader, desc=f"{model_name}-{name}"):
            if count >= num_samples: break
            images = images.to(wrapper.device)
            orig_pred = wrapper.predict(images)
            
            if callable(attack) and not hasattr(attack, 'attack'):
                adv = attack(wrapper, images, orig_pred)
            else:
                adv = attack.attack(images, orig_pred)
            
            if wrapper.predict(adv) != orig_pred:
                asr_count += 1
            
            # Metrics
            o_np = denorm_vision(images, is_cifar).squeeze(0).permute(1, 2, 0).cpu().numpy()
            a_np = denorm_vision(adv, is_cifar).squeeze(0).permute(1, 2, 0).cpu().numpy()
            try:
                s = ssim(o_np, a_np, channel_axis=2, data_range=1.0)
            except:
                s = ssim(o_np, a_np, multichannel=True, data_range=1.0)
            
            ssims.append(s)
            l2s.append(np.linalg.norm((o_np - a_np).flatten()))
            count += 1
            
        rows.append({
            "Modality": "Vision",
            "Model": model_name,
            "Method": name,
            "ASR": (asr_count/count)*100,
            "Quality (SSIM/PESQ)": np.mean(ssims),
            "L2 Energy": np.mean(l2s),
            "WER/Other": 0
        })
    return rows

def main():
    all_results = []
    
    # 1. Audio Benchmark (Sample size restricted to 50 for local time limits if needed, but trying 100)
    all_results.extend(run_audio_benchmark(num_samples=100))
    
    # 2. Vision Benchmark (CIFAR) - 100 samples
    cifar_loader = get_cifar10_loader(batch_size=1)
    for m in ["cifar10_resnet20", "cifar10_resnet56"]:
        all_results.extend(run_vision_benchmark(m, cifar_loader, is_cifar=True, num_samples=100))
        
    # 3. Vision Benchmark (Tiny-IN) - 100 samples
    tiny_loader = get_tiny_imagenet_loader(batch_size=1)
    for m in ["resnet101", "resnet152"]:
        all_results.extend(run_vision_benchmark(m, tiny_loader, is_cifar=False, num_samples=100))
        
    df = pd.DataFrame(all_results)
    df.to_csv("final_grand_results.csv", index=False)
    
    print("\n" + "="*100)
    print("GRAND UNIFIED MULTI-MODAL BENCHMARK (N=10, eps=fixed)")
    print(df.to_string())
    print("="*100)

if __name__ == "__main__":
    main()
