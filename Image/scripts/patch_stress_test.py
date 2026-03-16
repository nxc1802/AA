import sys
import os
import torch
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.pgd_patch_2d import PGDPatchAttack2D
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def main():
    model_name = "resnet152"
    model_wrapper = VisionModelWrapper(model_name=model_name)
    dataloader = get_tiny_imagenet_loader(batch_size=1)
    epsilon = 0.031
    num_samples = 30
    patch_percent = 0.1 # Very hard constraint
    
    # Attacks
    pgd_patch = PGDPatchAttack2D(model_wrapper, patch_percent=patch_percent, epsilon=epsilon, num_iter=10)
    hessian_patch = SubspaceHessianAttack2D(model_wrapper, patch_percent=patch_percent, num_iter=10, epsilon=epsilon)
    
    results = {"pgd": 0, "hessian": 0, "total": 0}
    
    count = 0
    for images, labels in tqdm(dataloader, desc=f"Patch Stress-Test {model_name}"):
        if count >= num_samples:
            break
            
        images = images.to(model_wrapper.device)
        orig_pred = model_wrapper.predict(images)
        
        # Untargeted: Misclassify relative to orig_pred
        adv_pgd = pgd_patch.attack(images, orig_pred)
        if model_wrapper.predict(adv_pgd) != orig_pred:
            results["pgd"] += 1
            
        adv_hess = hessian_patch.attack(images, orig_pred)
        if model_wrapper.predict(adv_hess) != orig_pred:
            results["hessian"] += 1
            
        results["total"] += 1
        count += 1
        
    print("\n" + "="*50)
    print(f"ULTRA-DEEP PATCH STRESS TEST (10% Area)")
    print(f"Model: {model_name}")
    print(f"PGD-Patch ASR:     {(results['pgd']/results['total'])*100:.2f}%")
    print(f"Hessian-Patch ASR: {(results['hessian']/results['total'])*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
