import sys
import os
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def main():
    print("[*] Verifying noise injection at tensor level...")
    wrapper = VisionModelWrapper(model_name="resnet152")
    loader = get_tiny_imagenet_loader(batch_size=1)
    
    images, labels = next(iter(loader))
    images = images.to(wrapper.device)
    orig_pred = wrapper.predict(images)
    epsilon = 0.031
    
    # 1. PGD
    adv_pgd = pgd_attack_2d(wrapper, images, orig_pred, epsilon=epsilon, num_iter=10)
    pgd_diff = (adv_pgd - images).abs().max().item()
    print(f"PGD Max Diff (Tensor):     {pgd_diff:.6f}")
    
    # 2. Hessian Patch
    hessian_patch = SubspaceHessianAttack2D(wrapper, patch_percent=0.1, num_iter=10, epsilon=epsilon)
    adv_hess = hessian_patch.attack(images, orig_pred)
    hess_diff = (adv_hess - images).abs().max().item()
    print(f"Hessian Max Diff (Tensor): {hess_diff:.6f}")
    
    # Check if they are bit-identical
    if pgd_diff == 0:
        print("[!!!] WARNING: PGD produced NO change.")
    if hess_diff == 0:
        print("[!!!] WARNING: Hessian produced NO change.")
        
    # Check predictions
    print(f"Original Pred: {orig_pred.item()}")
    print(f"PGD Pred:      {wrapper.predict(adv_pgd).item()}")
    print(f"Hessian Pred:  {wrapper.predict(adv_hess).item()}")

if __name__ == "__main__":
    main()
