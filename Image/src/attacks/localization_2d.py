import torch
import torch.nn.functional as F
import numpy as np

class PatchLocator2D:
    def __init__(self, patch_percent=0.1):
        self.patch_percent = patch_percent

    def find_patch(self, gradient):
        """
        Finds the H x W patch in a 2D image (C, H, W) that has the 
        highest cumulative absolute gradient magnitude.
        
        gradient: Tensor of shape (C, H, W) or (B, C, H, W)
        Returns: (y0, x0, ph, pw, mask)
        """
        if len(gradient.shape) == 4:
            gradient = gradient[0] # Take first item in batch
            
        C, H, W = gradient.shape
        
        # 1. Calculate patch dimensions
        patch_area = int(H * W * self.patch_percent)
        pw = ph = int(np.sqrt(patch_area))
        
        # 2. Compute absolute gradient magnitude summed across channels
        grad_mag = torch.sum(torch.abs(gradient), dim=0, keepdim=True) # (1, H, W)
        
        # Apply a small Gaussian blur to smooth gradient spikes and find robust regions
        # (Using a simple box filter as blur)
        smooth_grad = F.avg_pool2d(grad_mag.unsqueeze(0), kernel_size=3, stride=1, padding=1)
        
        # 3. Use 2D convolution with a ones-kernel to find the sum of magnitudes in every possible patch
        kernel = torch.ones((1, 1, ph, pw)).to(gradient.device)
        sums = F.conv2d(smooth_grad, kernel, padding=0) # (1, 1, H-ph+1, W-pw+1)
        
        # 4. Find the top-left index of the patch with the highest sum
        max_val, max_idx = torch.max(sums.view(-1), dim=0)
        
        # Convert flat index back to 2D
        y0 = int(max_idx.item() // sums.shape[3])
        x0 = int(max_idx.item() % sums.shape[3])
        
        # 5. Create binary mask
        mask = torch.zeros((1, C, H, W)).to(gradient.device)
        mask[:, :, y0:y0+ph, x0:x0+pw] = 1.0
        
        return y0, x0, ph, pw, mask

class PixelLocator2D:
    def __init__(self, top_k_percent=0.1):
        self.top_k_percent = top_k_percent

    def find_mask(self, gradient):
        """
        Creates a binary mask selecting the top K% pixels by absolute gradient.
        gradient: Tensor of shape (C, H, W) or (B, C, H, W)
        Returns: binary mask (1, C, H, W)
        """
        if len(gradient.shape) == 4:
            gradient = gradient[0] 
            
        C, H, W = gradient.shape
        # Sum absolute gradients across channels
        grad_mag = torch.sum(torch.abs(gradient), dim=0) # (H, W)
        
        # Flatten
        flat_grads = grad_mag.view(-1)
        k = int(len(flat_grads) * self.top_k_percent)
        if k == 0: k = 1 # At least one pixel
        
        # Find threshold for top K%
        threshold = torch.topk(flat_grads, k).values[-1]
        
        # Create binary mask
        mask = (grad_mag >= threshold).float()
        # Ensure exactly k pixels are selected (handling ties if necessary)
        # Actually torch.topk is fine. 
        # But (grad_mag >= threshold) might select slightly more than k if many have same value.
        # For simplicity in this research, this is acceptable.
        
        mask = mask.view(1, 1, H, W).repeat(1, C, 1, 1) # (1, C, H, W)
        return mask
