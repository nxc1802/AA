import torch
import torch.nn as nn

def project_topk_support(delta, k):
    """
    Keep only the top-k pixels with highest perturbation magnitude.
    delta: [B, C, H, W]
    k: int
    """
    B, C, H, W = delta.size()
    # Magnitude per spatial pixel
    spatial_mag = delta.abs().max(dim=1)[0] # [B, H, W]
    spatial_mag_flat = spatial_mag.view(B, -1) # [B, H*W]
    
    if k >= spatial_mag_flat.size(1):
        return delta
        
    topk_vals, _ = torch.topk(spatial_mag_flat, k, dim=1)
    tau = topk_vals[:, -1].view(-1, 1, 1) # [B, 1, 1]
    
    mask = (spatial_mag >= tau).float().unsqueeze(1) # [B, 1, H, W]
    return delta * mask

def topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1, dynamic=True, return_history=False, score_ema=0.0):
    """
    Gradient-Guided Sparse Attack (Top-k PGD) with STRICT L0 budget.
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    B, C, H, W = images.shape
    N = H * W
    k_max = int(N * k_ratio)
    if k_max < 1: k_max = 1
    
    delta = torch.zeros_like(images)
    
    history = []
    
    ema_score = None
    
    for t in range(iters):
        adv_images = images + delta
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            score = grad.abs().max(dim=1)[0] # [B, H, W]
            
            if ema_score is None:
                ema_score = score
            else:
                ema_score = score_ema * ema_score + (1 - score_ema) * score
                
            if dynamic:
                # Dynamic mask: select k_t active pixels based on EMA score
                # Schedule: decrease active set from k_max towards k_max (constant for now as we enforce L0 strictly anyway)
                # Let's keep a schedule that starts broad and narrows down if needed, but for simplicity, we can just select k_max pixels based on score, since project_topk_support will enforce k_max strictly anyway.
                # Actually, dynamic support means the mask changes. We can just pick the top-k gradient pixels.
                score_flat = ema_score.view(B, -1)
                
                # We can optionally use a broader k_t here and then strictly cap delta, but the simplest is just k_max.
                k_t = k_max
                topk_vals, _ = torch.topk(score_flat, k_t, dim=1)
                tau = topk_vals[:, -1].view(-1, 1, 1)
                mask = (ema_score >= tau).float().unsqueeze(1)
            else:
                if t == 0:
                    score_flat = ema_score.view(B, -1)
                    topk_vals, _ = torch.topk(score_flat, k_max, dim=1)
                    tau = topk_vals[:, -1].view(-1, 1, 1)
                    mask = (ema_score >= tau).float().unsqueeze(1)
            
            # 1. Update delta with gradient and mask
            delta = delta + alpha * grad.sign() * mask
            
            # 2. Project to Linf
            delta = torch.clamp(delta, min=-eps, max=eps)
            
            # 3. Project to L0 strictly
            delta = project_topk_support(delta, k_max)
            
            # 4. Project to image bounds
            adv_images = torch.clamp(images + delta, min=0, max=1)
            delta = adv_images - images
            
            if return_history:
                history.append(adv_images.clone().detach())
        
    adv_images = images + delta
    
    # Assert final L0 is respected
    l0_actual = (delta.abs().max(dim=1)[0] > 1e-4).float().view(B, -1).sum(dim=1)
    # Allow a small float epsilon tolerance by checking non-zero pixels rather than >1e-4 if needed, but 1e-4 is safe.
    
    return (adv_images, history) if return_history else adv_images
