import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_perturbation(image, adv_image, title="Perturbation"):
    image_np = image.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    adv_np = adv_image.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    
    diff = np.abs(adv_np - image_np)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Clean Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(adv_np)
    plt.title("Adversarial Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.title("Perturbation (Normalized)")
    plt.axis("off")
    
    plt.suptitle(title)
    plt.show()
