import sys
import os
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import get_cifar10_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.pgd_patch_2d import PGDPatchAttack2D
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def run_area_benchmark(model_wrapper, attack_class, area_percent, loader, num_samples=100, num_iter=10):
    print(f"[*] Testing Area: {area_percent*100}% | Attack: {attack_class.__name__}")
    
    # Initialize attack with specific area
    attack = attack_class(model_wrapper, patch_percent=area_percent, num_iter=num_iter, epsilon=0.031)
    
    successes = []
    count = 0
    for images, labels in tqdm(loader, desc=f"Area {area_percent*100}%"):
        if count >= num_samples:
            break
            
        orig_pred = model_wrapper.predict(images)
        if orig_pred == labels.to(model_wrapper.device):
            adv_images = attack.attack(images, labels)
            adv_pred = model_wrapper.predict(adv_images)
            success = (adv_pred != labels.to(model_wrapper.device)).item()
            successes.append(success)
            count += 1
            
    asr = (sum(successes) / len(successes)) * 100 if successes else 0
    return asr

def main():
    output_dir = "Image/outputs/tradeoff"
    os.makedirs(output_dir, exist_ok=True)
    
    model_wrapper = VisionModelWrapper(model_name="cifar10_resnet20")
    dataloader = get_cifar10_loader(batch_size=1)
    
    # Areas to sweep
    areas = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0] # 5% to 100%
    num_iter = 10
    num_samples = 100
    
    results = []
    
    for area in areas:
        # PGD Patch Sweep
        pgd_asr = run_area_benchmark(model_wrapper, PGDPatchAttack2D, area, dataloader, num_samples=num_samples, num_iter=num_iter)
        
        # Hessian Patch Sweep
        hess_asr = run_area_benchmark(model_wrapper, SubspaceHessianAttack2D, area, dataloader, num_samples=num_samples, num_iter=num_iter)
        
        results.append({
            "area_percent": area * 100,
            "pgd_asr": pgd_asr,
            "hessian_asr": hess_asr,
            "impact_density_pgd": pgd_asr / (area * 100 + 1e-8), # ASR per unit area
            "impact_density_hess": hess_asr / (area * 100 + 1e-8)
        })
        
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "area_tradeoff_results.csv"), index=False)
    
    print("\n" + "="*60)
    print(f"{'Area %':<10} | {'PGD ASR %':<12} | {'Hessian ASR %':<15} | {'PGD Density':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['area_percent']:<10.0f} | {r['pgd_asr']:<12.2f} | {r['hessian_asr']:<15.2f} | {r['impact_density_pgd']:<12.4f}")
    print("="*60)
    print(f"Full results saved to {output_dir}/area_tradeoff_results.csv")

if __name__ == "__main__":
    main()
