import sys
import os
import torch
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import get_cifar10_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.pgd_patch_2d import PGDPatchAttack2D
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def evaluate_efficiency(model_wrapper, attack_name, attack_fn, loader, num_samples=100):
    results = []
    count = 0
    for images, labels in tqdm(loader, desc=f"Efficiency {attack_name}"):
        if count >= num_samples:
            break
        orig_pred = model_wrapper.predict(images)
        if orig_pred == labels.to(model_wrapper.device):
            adv_images = attack_fn.attack(images, labels) if hasattr(attack_fn, 'attack') else attack_fn(model_wrapper, images, labels)
            adv_pred = model_wrapper.predict(adv_images)
            correct = (adv_pred == labels.to(model_wrapper.device)).item()
            results.append(correct)
            count += 1
    
    asr = (1 - sum(results)/len(results)) * 100 if results else 0
    return asr

def main():
    model_wrapper = VisionModelWrapper(model_name="cifar10_resnet20")
    dataloader = get_cifar10_loader(batch_size=1)
    epsilon = 0.031
    num_test = 100
    
    print("\n--- EFFICIENCY COMPARISON ---")
    
    # 1. Global Baseline (High Cost)
    print("[*] Running Global PGD (N=20) - High Cost...")
    pgd_global_20 = lambda m, i, l: pgd_attack_2d(m, i, l, epsilon=epsilon, num_iter=20)
    asr_global_20 = evaluate_efficiency(model_wrapper, "Global-PGD-20", pgd_global_20, dataloader, num_samples=num_test)
    
    # 2. Patch Baseline (Low Cost, Small Area)
    print(f"[*] Running PGD-Patch (N=10, 10% area) - Med Cost...")
    pgd_patch_10 = PGDPatchAttack2D(model_wrapper, patch_percent=0.1, epsilon=epsilon, num_iter=10)
    asr_patch_10 = evaluate_efficiency(model_wrapper, "Patch-10-N10", pgd_patch_10, dataloader, num_samples=num_test)
    
    # 3. Targeted Optimization (Ultra Low Cost, Smart Area)
    # Goal: Reach Global ASR with only 5 iterations!
    print(f"[*] Running PGD-Patch (N=5, 30% area) - Ultra Low Cost...")
    pgd_patch_5_s30 = PGDPatchAttack2D(model_wrapper, patch_percent=0.3, epsilon=epsilon, num_iter=5)
    asr_patch_5 = evaluate_efficiency(model_wrapper, "Patch-30-N5", pgd_patch_5_s30, dataloader, num_samples=num_test)
    
    # 4. Balanced Success (Med Cost, 50% area)
    print(f"[*] Running PGD-Patch (N=10, 50% area) - Targeted Efficiency...")
    pgd_patch_10_s50 = PGDPatchAttack2D(model_wrapper, patch_percent=0.5, epsilon=epsilon, num_iter=10)
    asr_patch_10_s50 = evaluate_efficiency(model_wrapper, "Patch-50-N10", pgd_patch_10_s50, dataloader, num_samples=num_test)

    print("\n" + "="*60)
    print(f"{'Attack':<25} | {'Cost (N)':<10} | {'Area':<10} | {'ASR %':<10}")
    print("-" * 60)
    print(f"{'Global PGD':<25} | {'20':<10} | {'100%':<10} | {asr_global_20:<10.2f}")
    print(f"{'PGD-Patch (v1)':<25} | {'10':<10} | {'10%':<10} | {asr_patch_10:<10.2f}")
    print(f"{'PGD-Patch (v2 - Fast)':<25} | {'5':<10} | {'30%':<10} | {asr_patch_5:<10.2f}")
    print(f"{'PGD-Patch (v3 - Preserved)':<25} | {'10':<10} | {'50%':<10} | {asr_patch_10_s50:<10.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
