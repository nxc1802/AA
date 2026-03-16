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
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def run_pure_bench(model_name, loader, num_samples=100, num_iter=5):
    print(f"\n[*] Evaluating Model Architecture: {model_name}")
    model_wrapper = VisionModelWrapper(model_name=model_name)
    epsilon = 0.031
    
    # Global Attacks (Area=100%, N=5)
    # 1. PGD (1st-order)
    pgd_attack = lambda m, i, l: pgd_attack_2d(m, i, l, epsilon=epsilon, num_iter=num_iter)
    # 2. Hessian (2nd-order)
    hessian_attack = SubspaceHessianAttack2D(model_wrapper, patch_percent=1.0, num_iter=num_iter, epsilon=epsilon)
    
    results = {"pgd_success": 0, "hess_success": 0, "total": 0}
    
    count = 0
    for images, labels in tqdm(loader, desc=f"Benchmarking {model_name}"):
        if count >= num_samples:
            break
            
        labels_dev = labels.to(model_wrapper.device)
        orig_pred = model_wrapper.predict(images)
        
        if orig_pred == labels_dev:
            # Run PGD
            adv_pgd = pgd_attack(model_wrapper, images, labels)
            pgd_pred = model_wrapper.predict(adv_pgd)
            if pgd_pred != labels_dev:
                results["pgd_success"] += 1
                
            # Run Hessian
            adv_hess = hessian_attack.attack(images, labels)
            hess_pred = model_wrapper.predict(adv_hess)
            if hess_pred != labels_dev:
                results["hess_success"] += 1
                
            results["total"] += 1
            count += 1
            
    pgd_asr = (results["pgd_success"] / results["total"]) * 100 if results["total"] > 0 else 0
    hess_asr = (results["hess_success"] / results["total"]) * 100 if results["total"] > 0 else 0
    
    return pgd_asr, hess_asr

def main():
    output_dir = "Image/outputs/complexity"
    os.makedirs(output_dir, exist_ok=True)
    
    dataloader = get_cifar10_loader(batch_size=1)
    
    # Scale of complexity (CIFAR-10 specific ResNets)
    models = ["cifar10_resnet20", "cifar10_resnet32", "cifar10_resnet44", "cifar10_resnet56"]
    num_samples = 100
    num_iter = 5 # Strict low-pass constraint where 2nd order should shine
    
    final_results = []
    
    for m in models:
        pgd_asr, hess_asr = run_pure_bench(m, dataloader, num_samples=num_samples, num_iter=num_iter)
        
        # Curvature Advantage
        gap = hess_asr - pgd_asr
        ratio = hess_asr / (pgd_asr + 1e-8)
        
        final_results.append({
            "model": m,
            "depth": int(m.split("resnet")[-1]),
            "pgd_asr": pgd_asr,
            "hessian_asr": hess_asr,
            "asr_gap": gap,
            "curvature_advantage": ratio
        })
        
    df = pd.DataFrame(final_results)
    df.to_csv(os.path.join(output_dir, "complexity_results.csv"), index=False)
    
    print("\n" + "="*70)
    print(f"{'Model':<20} | {'Depth':<6} | {'PGD ASR':<10} | {'Hessian ASR':<12} | {'Advantage'}")
    print("-" * 70)
    for r in final_results:
        print(f"{r['model']:<20} | {r['depth']:<6} | {r['pgd_asr']:<10.2f} | {r['hessian_asr']:<12.2f} | {r['curvature_advantage']:<10.3f}")
    print("="*70)
    print(f"Results saved to {output_dir}/complexity_results.csv")

if __name__ == "__main__":
    main()
