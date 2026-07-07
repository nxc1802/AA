import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_perturbation(image, adv_image, title="Perturbation", save_path=None):
    image_np = image.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    adv_np = adv_image.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    
    diff = np.abs(adv_np - image_np)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    # Compute active mask
    mask = (diff.sum(axis=-1) > 1e-5).astype(float)
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image_np)
    plt.title("Clean Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(adv_np)
    plt.title("Adversarial Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(diff_norm)
    plt.title("Perturbation (Normalized)")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Active Pixels: {int(mask.sum())}")
    plt.axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
