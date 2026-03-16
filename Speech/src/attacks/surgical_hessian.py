import torch
import numpy as np
from Speech.src.attacks.hessian_patch import forward_euler_vhp

class SurgicalHessianAttack:
    def __init__(self, model_wrapper, top_k_percent=0.1, num_iter=10, epsilon=0.005, is_targeted=False):
        self.model_wrapper = model_wrapper
        self.top_k_percent = top_k_percent
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.is_targeted = is_targeted
        from Speech.src.attacks.localization import PixelLocator
        self.locator = PixelLocator(top_k_percent=top_k_percent)

    def attack(self, audio_tensor, ground_truth, target_text=None):
        device = self.model_wrapper.device
        audio_tensor = audio_tensor.to(device)
        original_audio = audio_tensor.detach().clone()
        
        # Determine optimizing text
        opt_text = target_text if self.is_targeted else ground_truth
        
        # 1. Initial Gradient (g0)
        audio_x = audio_tensor.detach().clone().requires_grad_(True)
        loss, _ = self.model_wrapper.get_loss(audio_x, opt_text)
        self.model_wrapper.model.zero_grad()
        loss.backward()
        g0 = audio_x.grad.data.clone()
        
        # 2. Pixel-Wise Localization (Sparse Mask)
        mask = self.locator.find_mask(g0)
        
        # 3. Initialization
        perturbation = torch.zeros_like(original_audio)
        if self.is_targeted:
            perturbation = (-self.epsilon * g0.sign()) * mask
        else:
            perturbation = (self.epsilon * g0.sign()) * mask
        
        # 4. Frank-Wolfe Optimization
        for k in range(self.num_iter):
            v = perturbation.detach()
            
            # vHp = H * v
            vhp_estimate = forward_euler_vhp(self.model_wrapper, original_audio, opt_text, v, g0, mask=mask)
            
            # 2nd-order corrected gradient
            g_corrected = (g0 + vhp_estimate) * mask
            
            # Linear Minimization Oracle (LMO)
            if self.is_targeted:
                s = -self.epsilon * g_corrected.sign() * mask
            else:
                s = self.epsilon * g_corrected.sign() * mask
            
            # Step size
            gamma = 2.0 / (k + 2.0)
            
            # Update
            perturbation = (1 - gamma) * perturbation + gamma * s
            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
            
            # Projection
            adv_audio = torch.clamp(original_audio + perturbation, min=-1.0, max=1.0)
            perturbation = adv_audio - original_audio
            
        return (original_audio + perturbation).detach().cpu().numpy().squeeze()
