import torch
import torch.nn as nn
from Image.src.attacks.localization_2d import PatchLocator2D

class PGDPatchAttack2D:
    def __init__(self, model_wrapper, patch_percent=0.1, epsilon=0.031, alpha=0.01, num_iter=20):
        self.model_wrapper = model_wrapper
        self.patch_percent = patch_percent
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = model_wrapper.device
        self.locator = PatchLocator2D(patch_percent=patch_percent)

    def attack(self, image_tensor, label):
        """
        PGD attack constrained to a localized patch.
        """
        image_tensor = image_tensor.to(self.device)
        label = label.to(self.device)
        original_image = image_tensor.clone().detach()
        
        # 1. Find the optimal patch location using initial gradient
        temp_x = image_tensor.clone().detach().requires_grad_(True)
        loss, _ = self.model_wrapper.get_loss(temp_x, label)
        self.model_wrapper.zero_grad()
        loss.backward()
        initial_grad = temp_x.grad.data.clone()
        
        y0, x0, ph, pw, mask = self.locator.find_patch(initial_grad)
        
        # 2. Iterative PGD within the mask
        adv_image = image_tensor.clone().detach().requires_grad_(True)
        
        for i in range(self.num_iter):
            loss, _ = self.model_wrapper.get_loss(adv_image, label)
            self.model_wrapper.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Standard PGD step
                grad = adv_image.grad.sign()
                adv_image = adv_image + self.alpha * grad
                
                # Apply localized mask
                perturbation = adv_image - original_image
                perturbation = perturbation * mask # Constraint to subspace
                
                # Projection to epsilon ball
                perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
                adv_image = torch.clamp(original_image + perturbation, min=-2.5, max=2.5)
            
            adv_image = adv_image.detach().requires_grad_(True)
            self.model_wrapper.backward_count += 1
            
        return adv_image.detach()
