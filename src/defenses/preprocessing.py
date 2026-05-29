import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import BaseDefense

class MedianSmoothingDefense(BaseDefense):
    """
    Median filter defense to reduce high-frequency adversarial noise.
    Highly effective against very sparse (pixel-level) perturbations.
    """
    def __init__(self, kernel_size=3):
        super(MedianSmoothingDefense, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "Kernel size must be odd."

    def forward(self, x):
        # x shape: [b, c, h, w]
        b, c, h, w = x.size()
        padding = self.kernel_size // 2
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Unfold patches: [b, c, h, w, kernel_size, kernel_size]
        patches = x_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        patches = patches.contiguous().view(b, c, h, w, -1)
        
        # Take median along the unfolded dimensions
        median_val, _ = patches.median(dim=-1)
        return median_val

class BitReductionsDefense(BaseDefense):
    """
    Reduces color bit depth (quantization) to wipe out small adversarial perturbations.
    """
    def __init__(self, bits=3):
        super(BitReductionsDefense, self).__init__()
        self.levels = 2 ** bits

    def forward(self, x):
        return torch.round(x * (self.levels - 1)) / (self.levels - 1)

class JPEGCompressionDefense(BaseDefense):
    """
    Simulates JPEG compression to eliminate high-frequency adversarial noise
    by round-tripping tensors through the Pillow JPEG library.
    """
    def __init__(self, quality=75):
        super(JPEGCompressionDefense, self).__init__()
        self.quality = quality

    def forward(self, x):
        device = x.device
        b, c, h, w = x.size()
        from PIL import Image
        import io
        
        x_np = x.cpu().numpy()
        x_defended = []
        for i in range(b):
            # Convert channel-first [C, H, W] to PIL Image format [H, W, C]
            img_np = np.clip(x_np[i].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            # Save as JPEG in bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            
            # Open JPEG image and load
            img_dec = Image.open(buffer)
            dec_np = np.array(img_dec).astype(np.float32) / 255.0
            
            # Reshape back to channel-first [C, H, W]
            x_defended.append(dec_np.transpose(2, 0, 1))
            
        return torch.tensor(np.array(x_defended), device=device, dtype=x.dtype)

class RandomNoiseDefense(BaseDefense):
    """
    Injects small random Gaussian noise to disrupt the deliberate, selective alignment of sparse updates.
    """
    def __init__(self, std=0.02):
        super(RandomNoiseDefense, self).__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)

class RandomizedSmoothingModel(nn.Module):
    """
    Certified Robustness wrapper using Randomized Smoothing (Monte Carlo expectation-based voting).
    Adds Gaussian noise during inference to achieve smoothed consensus.
    Vectorized implementation for ultra-fast GPU/MPS execution.
    """
    def __init__(self, model, sigma=0.12, N=100):
        super(RandomizedSmoothingModel, self).__init__()
        self.model = model
        self.sigma = sigma
        self.N = N

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Vectorized Monte Carlo: duplicate inputs N times along batch dimension
        # x_expanded shape: [B * N, C, H, W]
        x_expanded = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(B * self.N, C, H, W)
        
        # Add random isotropic Gaussian noise
        noise = torch.randn_like(x_expanded) * self.sigma
        
        # Forward pass in one single vectorized batch
        logits = self.model(x_expanded + noise) # [B * N, NumClasses]
        
        # Reshape and average predictions across Monte Carlo samples
        logits = logits.view(B, self.N, -1).mean(dim=1)
        return logits

class FeatureDenoisingWrapper(nn.Module):
    """
    Zero-shot Feature Denoising defense that registers PyTorch forward hooks on intermediate 
    activations (e.g. layer2, layer3) of standard/robust ResNet models to filter adversarial noise.
    """
    def __init__(self, model, kernel_size=3):
        super(FeatureDenoisingWrapper, self).__init__()
        self.model = model
        self.kernel_size = kernel_size
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def denoise_hook(module, inp, out):
            # Apply spatial median filtering on the intermediate feature activation maps
            B, C, H, W = out.size()
            padding = self.kernel_size // 2
            out_pad = F.pad(out, (padding, padding, padding, padding), mode='reflect')
            # Extract spatial patches: [B, C, H, W, K, K]
            patches = out_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            patches = patches.contiguous().view(B, C, H, W, -1)
            median_val, _ = patches.median(dim=-1)
            return median_val

        # Hook standard ResNet stages
        for name, module in self.model.named_modules():
            if name in ['layer2', 'layer3']:
                h = module.register_forward_hook(denoise_hook)
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward(self, x):
        return self.model(x)

