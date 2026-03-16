import torch
import torch.nn.functional as F
import numpy as np

def forward_euler_vhp(model_wrapper, audio_tensor, text, v, g0, h=1e-3, mask=None):
    """
    Approximates Vector-Hessian Product (H * v) using Finite Differences (Forward Euler).
    H * v ≈ (grad(x + h*v) - grad(x)) / h
    
    g0: Pre-calculated gradient at audio_tensor (x).
    text: Can be ground truth (untargeted) or target text (targeted).
    mask: Binary mask to restrict computation to the subspace.
    """
    device = model_wrapper.device
    
    # Get g_h = grad(x + h*v)
    v_masked = v * mask if mask is not None else v
    audio_h = (audio_tensor.detach() + h * v_masked).clone().requires_grad_(True)
    
    loss_h, _ = model_wrapper.get_loss(audio_h, text)
    model_wrapper.model.zero_grad()
    loss_h.backward()
    model_wrapper.backward_count += 1
    gh = audio_h.grad.data.clone()
    
    # 2. vHp estimate
    vhp = (gh - g0) / h
    
    if mask is not None:
        vhp = vhp * mask
        
    return vhp

class SubspaceHessianAttack:
    def __init__(self, model_wrapper, patch_percent=0.1, step_strategy="decay", num_iter=10, epsilon=0.005, is_targeted=False):
        self.model_wrapper = model_wrapper
        self.patch_percent = patch_percent
        self.step_strategy = step_strategy
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.is_targeted = is_targeted
        from Speech.src.attacks.localization import PatchLocator
        self.locator = PatchLocator(patch_percent=patch_percent)

    def get_step_size(self, k, gradient, vhp=None, perturbation=None):
        if self.step_strategy == "decay":
            return 2.0 / (k + 2.0)
        elif self.step_strategy == "fixed":
            return 0.1
        elif self.step_strategy == "line_search":
            # Simplified line search for ASR: 
            # In a real scenario, this might involve multiple forward passes.
            # Placeholder: quadratic approximation or small backtracking.
            return 0.5 # Default fallback
        return 0.1

    def attack(self, audio_tensor, ground_truth, target_text=None):
        """
        ground_truth: The original label (used for untargeted).
        target_text: The goal label (used for targeted).
        """
        device = self.model_wrapper.device
        audio_tensor = audio_tensor.to(device)
        original_audio = audio_tensor.detach().clone()
        
        # Determine optimizing text
        opt_text = target_text if self.is_targeted else ground_truth
        
        # Step 1: Global Gradient (g0)
        audio_x = audio_tensor.detach().clone().requires_grad_(True)
        loss, _ = self.model_wrapper.get_loss(audio_x, opt_text)
        self.model_wrapper.model.zero_grad()
        loss.backward()
        self.model_wrapper.backward_count += 1
        g0 = audio_x.grad.data.clone()
        
        # Step 2: Localization (always based on g0)
        start_idx, end_idx, mask = self.locator.find_patch(g0)
        
        # Step 3: Initialization
        perturbation = torch.zeros_like(original_audio)
        if self.is_targeted:
            # For targeted, we move AGAINST g0 to minimize loss to target
            perturbation = (-self.epsilon * g0.sign()) * mask
        else:
            # For untargeted, we move WITH g0 to maximize loss from ground truth
            perturbation = (self.epsilon * g0.sign()) * mask
        
        # Step 4: Frank-Wolfe Optimization
        for k in range(self.num_iter):
            current_audio = (original_audio + perturbation).detach()
            v = perturbation.detach()
            
            # vHp = H * v (approximated based on opt_text)
            vhp_estimate = forward_euler_vhp(self.model_wrapper, original_audio, opt_text, v, g0, mask=mask)
            
            # 2nd-order corrected gradient
            g_corrected = (g0 + vhp_estimate) * mask
            
            # Linear Minimization Oracle (LMO)
            if self.is_targeted:
                # Minimize loss to target: s = -epsilon * sign(g_corrected)
                s = -self.epsilon * g_corrected.sign() * mask
            else:
                # Maximize loss from ground truth: s = epsilon * sign(g_corrected)
                s = self.epsilon * g_corrected.sign() * mask
            
            # Step size
            gamma = self.get_step_size(k, g_corrected)
            
            # Update
            perturbation = (1 - gamma) * perturbation + gamma * s
            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
            
            # Projection
            adv_audio = torch.clamp(original_audio + perturbation, min=-1.0, max=1.0)
            perturbation = adv_audio - original_audio
            
        return (original_audio + perturbation).detach().cpu().numpy().squeeze()
