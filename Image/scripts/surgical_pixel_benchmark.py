import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D
from src.attacks.surgical_hessian_2d import SurgicalHessianAttack2D

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def get_metrics(orig, adv):
    orig_np = denormalize(orig).squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = denormalize(adv).squeeze(0).permute(1, 2, 0).cpu().numpy()
    try:
        s = ssim(orig_np, adv_np, channel_axis=2, data_range=1.0)
    except:
        s = ssim(orig_np, adv_np, multichannel=True, data_range=1.0)
    l2 = np.linalg.norm((orig_np - adv_np).flatten())
    return s, l2

def main():
    model_name = "resnet152"
    wrapper = VisionModelWrapper(model_name=model_name)
    dataloader = get_tiny_imagenet_loader(batch_size=1)
    epsilon = 0.031
    num_samples = 50
    area_percent = 0.1
    
    # Attacks
    patch_attack = SubspaceHessianAttack2D(wrapper, patch_percent=area_percent, num_iter=10, epsilon=epsilon)
    surgical_attack = SurgicalHessianAttack2D(wrapper, top_k_percent=area_percent, num_iter=10, epsilon=epsilon)
    
    results = []
    
    for images, labels in tqdm(dataloader, desc="Surgical vs Patch"):
        if len(results) >= num_samples: break
        
        images = images.to(wrapper.device)
        orig_pred = wrapper.predict(images)
        
        # 1. Patch Attack
        adv_patch = patch_attack.attack(images, orig_pred)
        patch_asr = 1 if wrapper.predict(adv_patch) != orig_pred else 0
        p_ssim, p_l2 = get_metrics(images, adv_patch)
        
        # 2. Surgical (Pixel-wise) Attack
        adv_surg = surgical_attack.attack(images, orig_pred)
        surg_asr = 1 if wrapper.predict(adv_surg) != orig_pred else 0
        s_ssim, s_l2 = get_metrics(images, adv_surg)
        
        results.append({
            "patch_asr": patch_asr,
            "patch_ssim": p_ssim,
            "patch_l2": p_l2,
            "surg_asr": surg_asr,
            "surg_ssim": s_ssim,
            "surg_l2": s_l2
        })
        
    df = pd.DataFrame(results)
    df.to_csv("Image/outputs/complexity/surgical_pixel_results.csv", index=False)
    
    print("\n" + "="*80)
    print(f"SURGICAL PIXEL vs RECTANGULAR PATCH (Area={area_percent*100}%, ResNet-152)")
    print(f"{'Metric':<15} | {'Patch (Box)':<15} | {'Surgical (Pixel)'}")
    print("-" * 80)
    print(f"{'ASR':<15} | {df['patch_asr'].mean()*100:<15.1f}% | {df['surg_asr'].mean()*100:.1f}%")
    print(f"{'Avg SSIM':<15} | {df['patch_ssim'].mean():<15.4f} | {df['surg_ssim'].mean():.4f}")
    print(f"{'Avg L2':<15} | {df['patch_l2'].mean():<15.4f} | {df['surg_l2'].mean():.4f}")
    print("="*80)
    
    asr_gain = (df['surg_asr'].mean() / (df['patch_asr'].mean() + 1e-8) - 1) * 100
    print(f"Strategic ASR Gain: {asr_gain:.2f}%")

if __name__ == "__main__":
    main()
