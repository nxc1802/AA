import torch
import torch.nn as nn
import numpy as np

def forward_euler_vhp_2d(model_wrapper, x, y_true, v, g0, steps=1, h=1e-3):
    """
    Approximates H*v using Forward Euler (finite difference of gradients).
    H*v = (g(x + h*v) - g(x)) / h
    Optimized for 2D Image Tensors (B, C, H, W).
    """
    device = model_wrapper.device
    v = v.to(device)
    g0 = g0.to(device)
    
    # 1. Perturb the input in the direction of v
    x_plus_hv = x.detach().to(device) + h * v
    x_plus_hv.requires_grad = True
    
    # 2. Get gradient at x + h*v
    loss, _ = model_wrapper.get_loss(x_plus_hv, y_true)
    model_wrapper.zero_grad()
    loss.backward()
    
    g_hv = x_plus_hv.grad
    model_wrapper.backward_count += 1
    
    # 3. Finite difference
    vhp = (g_hv - g0) / h
    return vhp

class SubspaceHessianAttack2D:
    def __init__(self, model_wrapper, patch_percent=0.1, num_iter=10, epsilon=0.005, is_targeted=False):
        self.model_wrapper = model_wrapper
        self.patch_percent = patch_percent
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.is_targeted = is_targeted
        self.device = model_wrapper.device
        from Image.src.attacks.localization_2d import PatchLocator2D
        self.locator = PatchLocator2D(patch_percent=patch_percent)

    def attack(self, image_tensor, label, target_label=None):
        """
        Subspace Hessian-Patch attack for images.
        N+1 Pass Optimization Strategy.
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
        
        # 1. Initial Gradient (g0) - The "+1" pass
        audio_x = image_tensor.clone().requires_grad_(True)
        loss, _ = self.model_wrapper.get_loss(audio_x, opt_label)
        self.model_wrapper.zero_grad()
        loss.backward()
        g0 = audio_x.grad.data.clone()
        
        # 2. Dynamic 2D Localization using PatchLocator2D
        y0, x0, ph, pw, mask = self.locator.find_patch(g0)
        
        # 3. Initialization based on Target/Untargeted
        if self.is_targeted:
            # Minimize loss to target
            perturbation = (-self.epsilon * g0.sign()) * mask
        else:
            # Maximize loss from ground truth
            perturbation = (self.epsilon * g0.sign()) * mask
        
        # 4. Frank-Wolfe Optimization (N passes)
        for k in range(self.num_iter):
            v = perturbation.detach()
            
            # vHp = H * v (approximated based on opt_label)
            vhp_estimate = forward_euler_vhp_2d(self.model_wrapper, original_image, opt_label, v, g0)
            
            # 2nd-order corrected gradient: g_step = g0 + H*v
            g_corrected = (g0 + vhp_estimate) * mask
            
            # LMO: Linear Minimization Oracle
            if self.is_targeted:
                # Minimize loss: move against g_corrected
                s = -self.epsilon * g_corrected.sign() * mask
            else:
                # Maximize loss: move with g_corrected
                s = self.epsilon * g_corrected.sign() * mask
            
            # Step size (decay)
            gamma = 2.0 / (k + 2.0)
            
            # Update
            perturbation = (1 - gamma) * perturbation + gamma * s
            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
            
            # Constraint and range enforcement
            adv_image = torch.clamp(original_image + perturbation, min=-2.5, max=2.5)
            perturbation = adv_image - original_image
            
        return (original_image + perturbation).detach()
