import torch
import torch.nn as nn
import numpy as np
from Image.src.attacks.hessian_patch_2d import forward_euler_vhp_2d

class SurgicalHessianAttack2D:
    def __init__(self, model_wrapper, top_k_percent=0.1, num_iter=10, epsilon=0.005, is_targeted=False):
        self.model_wrapper = model_wrapper
        self.top_k_percent = top_k_percent
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.is_targeted = is_targeted
        self.device = model_wrapper.device
        from Image.src.attacks.localization_2d import PixelLocator2D
        self.locator = PixelLocator2D(top_k_percent=top_k_percent)

    def attack(self, image_tensor, label, target_label=None):
        """
        Surgical Pixel Hessian attack.
        Optimizes only the top K% influential pixels.
        """
        B, C, H, W = image_tensor.shape
        image_tensor = image_tensor.to(self.device).detach()
        original_image = image_tensor.clone()
        
        # Determine optimizing goal
        if self.is_targeted:
            opt_label = target_label if target_label is not None else label
        else:
            opt_label = label
        opt_label = opt_label.to(self.device)
        
        # 1. Initial Gradient (g0)
        audio_x = image_tensor.clone().requires_grad_(True)
        loss, _ = self.model_wrapper.get_loss(audio_x, opt_label)
        self.model_wrapper.zero_grad()
        loss.backward()
        g0 = audio_x.grad.data.clone()
        
        # 2. Pixel-Wise Localization (Sparse Mask)
        mask = self.locator.find_mask(g0)
        
        # 3. Initialization
        if self.is_targeted:
            perturbation = (-self.epsilon * g0.sign()) * mask
        else:
            perturbation = (self.epsilon * g0.sign()) * mask
        
        # 4. Frank-Wolfe Optimization
        for k in range(self.num_iter):
            v = perturbation.detach()
            
            # vHp = H * v
            vhp_estimate = forward_euler_vhp_2d(self.model_wrapper, original_image, opt_label, v, g0)
            
            # 2nd-order corrected gradient
            g_corrected = (g0 + vhp_estimate) * mask
            
            # LMO: Linear Minimization Oracle
            if self.is_targeted:
                s = -self.epsilon * g_corrected.sign() * mask
            else:
                s = self.epsilon * g_corrected.sign() * mask
            
            # Step size
            gamma = 2.0 / (k + 2.0)
            
            # Update
            perturbation = (1 - gamma) * perturbation + gamma * s
            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
            
            # Constraint enforcement
            adv_image = torch.clamp(original_image + perturbation, min=-2.5, max=2.5)
            perturbation = adv_image - original_image
            
        return (original_image + perturbation).detach()
