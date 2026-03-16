import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import get_cifar10_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def denormalize(tensor):
    """
    Inverse CIFAR-10 normalization for visualization.
    """
    device = tensor.device
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    res = tensor * std + mean
    return torch.clamp(res, 0, 1)

def main():
    os.makedirs("Image/outputs/plots", exist_ok=True)
    
    # Setup
    print("[*] Loading Model and Data...")
    model_wrapper = VisionModelWrapper(model_name="cifar10_resnet20")
    dataloader = get_cifar10_loader(batch_size=1)
    
    # Get one valid sample
    image, label = next(iter(dataloader))
    while model_wrapper.predict(image).item() != label.item():
        image, label = next(iter(dataloader))
        
    # Attack 1: Subspace (10%)
    print("[*] Running Subspace Attack...")
    attack_sub = SubspaceHessianAttack2D(model_wrapper, patch_percent=0.1, num_iter=10, epsilon=0.031)
    adv_sub = attack_sub.attack(image, label)
    
    # Attack 2: Global (100%)
    print("[*] Running Global Attack...")
    attack_global = SubspaceHessianAttack2D(model_wrapper, patch_percent=1.0, num_iter=10, epsilon=0.031)
    adv_global = attack_global.attack(image, label)
    
    # Prepare for plotting
    img_orig = denormalize(image[0]).permute(1, 2, 0).cpu().numpy()
    img_adv_sub = denormalize(adv_sub[0]).permute(1, 2, 0).cpu().numpy()
    img_adv_global = denormalize(adv_global[0]).permute(1, 2, 0).cpu().numpy()
    
    noise_sub = np.abs(img_adv_sub - img_orig)
    noise_sub = noise_sub / (noise_sub.max() + 1e-8) # Normalize for visibility
    
    # Metrics
    ssim_sub = ssim(img_orig, img_adv_sub, channel_axis=2, data_range=1.0)
    ssim_global = ssim(img_orig, img_adv_global, channel_axis=2, data_range=1.0)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_orig)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(img_adv_sub)
    axes[1].set_title(f"Subspace Attack (10%)\nSSIM: {ssim_sub:.4f}")
    axes[1].axis('off')
    
    axes[2].imshow(noise_sub)
    axes[2].set_title("Localized Noise (10%)")
    axes[2].axis('off')
    
    axes[3].imshow(img_adv_global)
    axes[3].set_title(f"Global Attack (100%)\nSSIM: {ssim_global:.4f}")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig("Image/outputs/plots/attack_visualization.png")
    print(f"[*] Visualization saved to Image/outputs/plots/attack_visualization.png")

if __name__ == "__main__":
    main()
