import sys
import os
import torch
import pandas as pd
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import get_cifar10_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import fgsm_attack_2d, pgd_attack_2d
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D
from src.attacks.pgd_patch_2d import PGDPatchAttack2D

def run_vision_benchmark(model_wrapper, attack_name, attack_fn, loader, num_samples=100):
    print(f"[*] Benchmarking {attack_name}...")
    results = []
    
    count = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {attack_name}"):
        if count >= num_samples:
            break
            
        model_wrapper.forward_count = 0
        model_wrapper.backward_count = 0
        
        # Original prediction
        orig_pred = model_wrapper.predict(images)
        
        # Execute Attack
        if hasattr(attack_fn, 'attack'):
            adv_images = attack_fn.attack(images, labels)
        else:
            adv_images = attack_fn(model_wrapper, images, labels)
            
        # Adversarial prediction
        adv_pred = model_wrapper.predict(adv_images)
        
        # Calculate stats
        correct = (adv_pred == labels.to(model_wrapper.device)).item()
        was_correct_originally = (orig_pred == labels.to(model_wrapper.device)).item()
        
        results.append({
            "id": count,
            "orig_correct": was_correct_originally,
            "adv_correct": correct,
            "fw_passes": model_wrapper.forward_count,
            "bw_passes": model_wrapper.backward_count
        })
        
        count += 1
        
    df = pd.DataFrame(results)
    # Only count samples that were originally correct
    df_valid = df[df["orig_correct"] == True]
    
    summary = {
        "accuracy_orig": float(df["orig_correct"].mean() * 100),
        "accuracy_adv": float(df["adv_correct"].mean() * 100),
        "asr": float((1 - df_valid["adv_correct"].mean()) * 100) if not df_valid.empty else 0.0,
        "avg_fw": float(df["fw_passes"].mean()),
        "avg_bw": float(df["bw_passes"].mean())
    }
    return summary

def main():
    output_dir = "Image/outputs/benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading CIFAR-10...")
    dataloader = get_cifar10_loader(batch_size=1)
    
    print("Loading CIFAR-10 ResNet-20...")
    model_wrapper = VisionModelWrapper(model_name="cifar10_resnet20")
    
    # Attacks
    hessian_sub = SubspaceHessianAttack2D(model_wrapper, patch_percent=0.1, num_iter=10, epsilon=0.031)
    hessian_global = SubspaceHessianAttack2D(model_wrapper, patch_percent=1.0, num_iter=10, epsilon=0.031)
    pgd_patch = PGDPatchAttack2D(model_wrapper, patch_percent=0.1, epsilon=0.031, num_iter=10) # 10 iterations to match Hessian
    fgsm = fgsm_attack_2d
    pgd = lambda m, i, l: pgd_attack_2d(m, i, l, epsilon=0.031, num_iter=10) # Set to 10 for fair comparison
    
    results = {}
    results["Hessian-Subspace"] = run_vision_benchmark(model_wrapper, "Hessian-Subspace", hessian_sub, dataloader, num_samples=100)
    results["Hessian-Global"] = run_vision_benchmark(model_wrapper, "Hessian-Global", hessian_global, dataloader, num_samples=100)
    results["PGD-Patch"] = run_vision_benchmark(model_wrapper, "PGD-Patch", pgd_patch, dataloader, num_samples=100)
    results["FGSM"] = run_vision_benchmark(model_wrapper, "FGSM", fgsm, dataloader, num_samples=100)
    results["PGD-10"] = run_vision_benchmark(model_wrapper, "PGD-10", pgd, dataloader, num_samples=100)
    
    with open(os.path.join(output_dir, "cifar10_resnet18_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*50)
    print("IMAGE BENCHMARK (CIFAR-10) COMPLETE")
    print(f"Results saved to {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
