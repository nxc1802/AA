import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def denormalize(tensor):
    """Denormalize ImageNet tensors to [0, 1] for SSIM."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def measure_perceptual(orig, adv):
    """Calculate SSIM, MSE, and L2."""
    orig_np = denormalize(orig).squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = denormalize(adv).squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # SSIM (multichannel=True is deprecated in newer skimage, using channel_axis)
    try:
        s = ssim(orig_np, adv_np, channel_axis=2, data_range=1.0)
    except:
        s = ssim(orig_np, adv_np, multichannel=True, data_range=1.0)
        
    mse = np.mean((orig_np - adv_np)**2)
    l2 = np.linalg.norm((orig_np - adv_np).flatten())
    
    return s, mse, l2

def main():
    model_name = "resnet152"
    wrapper = VisionModelWrapper(model_name=model_name)
    dataloader = get_tiny_imagenet_loader(batch_size=1)
    epsilon = 0.031
    num_samples = 30
    
    # 1. Global PGD (100% area)
    # 2. Hessian-Patch (10% area)
    hessian_patch = SubspaceHessianAttack2D(wrapper, patch_percent=0.1, num_iter=10, epsilon=epsilon)
    
    results = []
    
    for images, labels in tqdm(dataloader, desc="Perceptual Comparison"):
        if len(results) >= num_samples: break
        
        images = images.to(wrapper.device)
        orig_pred = wrapper.predict(images)
        
        # PGD Attack
        adv_pgd = pgd_attack_2d(wrapper, images, orig_pred, epsilon=epsilon, num_iter=10)
        pgd_ssim, pgd_mse, pgd_l2 = measure_perceptual(images, adv_pgd)
        
        # Hessian Patch Attack
        adv_hess = hessian_patch.attack(images, orig_pred)
        hess_ssim, hess_mse, hess_l2 = measure_perceptual(images, adv_hess)
        
        results.append({
            "pgd_ssim": pgd_ssim,
            "pgd_l2": pgd_l2,
            "hess_ssim": hess_ssim,
            "hess_l2": hess_l2
        })
        
    df = pd.DataFrame(results)
    df.to_csv("Image/outputs/complexity/perceptual_results.csv", index=False)
    
    print("\n" + "="*50)
    print(f"PERCEPTUAL DISTORTION COMPARISON (ResNet-152)")
    print(f"{'Metric':<15} | {'Global PGD':<15} | {'Hessian-Patch 10%'}")
    print("-" * 50)
    print(f"{'Avg SSIM':<15} | {df['pgd_ssim'].mean():<15.4f} | {df['hess_ssim'].mean():.4f}")
    print(f"{'Avg L2 (Energy)':<15} | {df['pgd_l2'].mean():<15.4f} | {df['hess_l2'].mean():.4f}")
    print("="*50)
    
    # Calculate Distrotion Reduction
    reduction = (1 - df['hess_l2'].mean() / df['pgd_l2'].mean()) * 100
    print(f"Distortion Energy Reduction: {reduction:.2f}%")

if __name__ == "__main__":
    main()
