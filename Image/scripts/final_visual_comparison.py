import sys
import os
import torch
import numpy as np
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def save_img(tensor, path):
    img = denormalize(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    return img

def save_noise_scaled(orig_tensor, adv_tensor, path):
    # Normalized noise map (magnified for human eyes)
    diff = (adv_tensor - orig_tensor).abs().squeeze(0).permute(1, 2, 0).cpu().numpy()
    diff = diff / (diff.max() + 1e-8)
    diff = (diff * 255).astype(np.uint8)
    Image.fromarray(diff).save(path)

def main():
    output_dir = "Image/outputs/visuals_resnet152"
    os.makedirs(output_dir, exist_ok=True)
    
    wrapper = VisionModelWrapper(model_name="resnet152")
    loader = get_tiny_imagenet_loader(batch_size=1)
    
    images, labels = next(iter(loader))
    images = images.to(wrapper.device)
    orig_pred = wrapper.predict(images)
    
    # We use a slightly LARGER epsilon (0.05 instead of 0.031) to make it more visible for the user
    epsilon = 0.05 
    
    # 1. Global PGD
    adv_pgd = pgd_attack_2d(wrapper, images, orig_pred, epsilon=epsilon, num_iter=10)
    
    # 2. Hessian Patch (10%)
    hessian_patch = SubspaceHessianAttack2D(wrapper, patch_percent=0.1, num_iter=10, epsilon=epsilon)
    adv_hess = hessian_patch.attack(images, orig_pred)
    
    # Save with unique names to avoid caching
    orig_np = save_img(images, f"{output_dir}/viz_original.png")
    pgd_np = save_img(adv_pgd, f"{output_dir}/viz_pgd_adv.png")
    hess_np = save_img(adv_hess, f"{output_dir}/viz_hessian_adv.png")
    
    save_noise_scaled(images, adv_pgd, f"{output_dir}/viz_pgd_noise_map.png")
    save_noise_scaled(images, adv_hess, f"{output_dir}/viz_hessian_noise_map.png")
    
    # PIXEL LEVEL COMPARISON
    pgd_pixel_diff = np.max(np.abs(orig_np.astype(float) - pgd_np.astype(float)))
    hess_pixel_diff = np.max(np.abs(orig_np.astype(float) - hess_np.astype(float)))
    
    print(f"[*] Visual Audit Results:")
    print(f"PGD Max Pixel Change (0-255):     {pgd_pixel_diff}")
    print(f"Hessian Max Pixel Change (0-255): {hess_pixel_diff}")
    
    if pgd_pixel_diff > 0 and hess_pixel_diff > 0:
        print("[OK] Noise is successfully baked into the PNG files.")
    else:
        print("[FAIL] Noise was lost during PNG encoding.")

if __name__ == "__main__":
    main()
