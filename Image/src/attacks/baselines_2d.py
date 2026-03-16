import torch
import torch.nn as nn

def fgsm_attack_2d(model_wrapper, image_tensor, label, epsilon=0.031):
    """
    Standard FGSM for images.
    """
    image_tensor = image_tensor.clone().detach().to(model_wrapper.device).requires_grad_(True)
    
    loss, _ = model_wrapper.get_loss(image_tensor, label)
    model_wrapper.zero_grad()
    loss.backward()
    model_wrapper.backward_count += 1
    
    with torch.no_grad():
        perturbed_image = image_tensor + epsilon * image_tensor.grad.sign()
        # Common ImageNet normalization clamp is roughly [-2.1, 2.7] or [0, 1] before norm
        # We will clamp to a reasonable range or based on the input stats
        perturbed_image = torch.clamp(perturbed_image, min=-2.5, max=2.5)
        
    return perturbed_image.detach()

def pgd_attack_2d(model_wrapper, image_tensor, label, epsilon=0.031, alpha=0.0078, num_iter=20):
    """
    Standard PGD for images.
    """
    original_image = image_tensor.clone().detach().to(model_wrapper.device)
    perturbed_image = original_image.clone().detach().requires_grad_(True)
    
    for i in range(num_iter):
        perturbed_image.requires_grad = True
        loss, _ = model_wrapper.get_loss(perturbed_image, label)
        
        model_wrapper.zero_grad()
        loss.backward()
        model_wrapper.backward_count += 1
        
        with torch.no_grad():
            adv_image = perturbed_image + alpha * perturbed_image.grad.sign()
            eta = torch.clamp(adv_image - original_image, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(original_image + eta, min=-2.5, max=2.5)
            
        perturbed_image = perturbed_image.detach()
        
    return perturbed_image
