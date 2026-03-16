import sys
import os
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def run_targeted_extreme_bench(model_name, loader, num_samples=30, num_iter=10):
    print(f"\n[*] Evaluating Targeted Ultra-Deep: {model_name}")
    model_wrapper = VisionModelWrapper(model_name=model_name)
    epsilon = 0.031
    target_class = 1 # Goldfish
    
    # Targeted Attacks
    hessian_targeted = SubspaceHessianAttack2D(model_wrapper, patch_percent=1.0, num_iter=num_iter, epsilon=epsilon, is_targeted=True)
    
    # Standard PGD doesn't have a targeted mode in my baselines_2d, let's implement a quick targeted PGD
    def targeted_pgd(wrapper, x, target_idx, eps, steps):
        x = x.clone().detach().requires_grad_(True)
        target = torch.tensor([target_idx]).to(wrapper.device)
        alpha = eps / 5
        for _ in range(steps):
            loss, _ = wrapper.get_loss(x, target)
            wrapper.zero_grad()
            loss.backward()
            with torch.no_grad():
                # For targeted, we MINIMIZE loss to target
                x = x - alpha * x.grad.sign()
                x = torch.clamp(x, min=-2.5, max=2.5) # Rough bound
            x = x.detach().requires_grad_(True)
        return x.detach()

    results = {"pgd_success": 0, "hess_success": 0, "total": 0}
    
    count = 0
    for images, labels in tqdm(loader, desc=f"Targeted {model_name}"):
        if count >= num_samples:
            break
            
        images = images.to(model_wrapper.device)
        orig_pred = model_wrapper.predict(images)
        
        if orig_pred != target_class:
            # Run Targeted PGD
            adv_pgd = targeted_pgd(model_wrapper, images, target_class, epsilon, num_iter)
            pgd_pred = model_wrapper.predict(adv_pgd)
            if pgd_pred == target_class:
                results["pgd_success"] += 1
                
            # Run Targeted Hessian
            adv_hess = hessian_targeted.attack(images, torch.tensor([target_class]))
            hess_pred = model_wrapper.predict(adv_hess)
            if hess_pred == target_class:
                results["hess_success"] += 1
                
            results["total"] += 1
            count += 1
            
    pgd_asr = (results["pgd_success"] / results["total"]) * 100 if results["total"] > 0 else 0
    hess_asr = (results["hess_success"] / results["total"]) * 100 if results["total"] > 0 else 0
    
    return pgd_asr, hess_asr

def main():
    output_dir = "Image/outputs/complexity"
    os.makedirs(output_dir, exist_ok=True)
    
    dataloader = get_tiny_imagenet_loader(batch_size=1)
    models = ["resnet50", "resnet101", "resnet152"]
    num_samples = 30 
    num_iter = 10
    
    final_results = []
    
    for m in models:
        pgd_asr, hess_asr = run_targeted_extreme_bench(m, dataloader, num_samples=num_samples, num_iter=num_iter)
        
        final_results.append({
            "model": m,
            "pgd_asr": pgd_asr,
            "hessian_asr": hess_asr,
            "curvature_advantage": hess_asr / (pgd_asr + 1e-8)
        })
        
    df = pd.DataFrame(final_results)
    df.to_csv(os.path.join(output_dir, "targeted_extreme_results.csv"), index=False)
    
    print("\n" + "="*70)
    print(f"{'Model':<20} | {'PGD T-ASR':<10} | {'Hessian T-ASR':<12} | {'Advantage'}")
    print("-" * 70)
    for r in final_results:
        print(f"{r['model']:<20} | {r['pgd_asr']:<10.2f} | {r['hessian_asr']:<12.2f} | {r['curvature_advantage']:<10.3f}")
    print("="*70)

if __name__ == "__main__":
    main()
