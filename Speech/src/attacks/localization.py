import torch
import numpy as np

class PatchLocator:
    def __init__(self, patch_percent=0.1):
        """
        patch_percent: Percentage of the audio length to be used as a patch.
        """
        self.patch_percent = patch_percent

    def find_patch(self, gradient):
        """
        gradient: 1D torch tensor or numpy array of the global gradient.
        Returns: (start_idx, end_idx, mask)
        """
        if isinstance(gradient, torch.Tensor):
            g = gradient.detach().cpu().abs().numpy().flatten()
        else:
            g = np.abs(gradient).flatten()
            
        N = len(g)
        L = int(N * self.patch_percent)
        
        if L <= 0:
            L = 1
        if L > N:
            L = N
            
        # O(N) Sliding Window using Prefix Sums
        prefix_sum = np.zeros(N + 1)
        prefix_sum[1:] = np.cumsum(g)
        
        # window_sum[i] = prefix_sum[i+L] - prefix_sum[i]  (sum of g[i : i+L])
        window_sums = prefix_sum[L:] - prefix_sum[:-L]
        
        max_idx = np.argmax(window_sums)
        start_idx = max_idx
        end_idx = max_idx + L
        
        # Create 1D mask
        mask = torch.zeros(N)
        mask[start_idx:end_idx] = 1.0
        
        return start_idx, end_idx, mask.to(gradient.device) if isinstance(gradient, torch.Tensor) else mask

class PixelLocator:
    def __init__(self, top_k_percent=0.1):
        self.top_k_percent = top_k_percent

    def find_mask(self, gradient):
        """
        Creates a binary mask selecting the top K% samples by absolute magnitude.
        """
        if isinstance(gradient, torch.Tensor):
            g = gradient.detach().abs().flatten()
        else:
            g = np.abs(gradient).flatten()
            
        N = len(g)
        k = int(N * self.top_k_percent)
        if k == 0: k = 1
        
        # Find threshold using top-k
        if isinstance(gradient, torch.Tensor):
            threshold = torch.topk(g, k).values[-1]
            mask = (g >= threshold).float()
        else:
            threshold = np.partition(g, -k)[-k]
            mask = (g >= threshold).astype(np.float32)
            
        return mask.view_as(gradient) if isinstance(gradient, torch.Tensor) else mask.reshape(gradient.shape)
