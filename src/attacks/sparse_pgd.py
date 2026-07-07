import torch
import torch.nn as nn

def project_l0_coordinates(delta, k):
    """
    Project delta to L0 ball with k non-zero elements (over all coordinates).
    """
    B = delta.size(0)
    delta_flat = delta.view(B, -1)
    if k >= delta_flat.size(1):
        return delta
        
    mag = delta_flat.abs()
    topk_vals, _ = torch.topk(mag, k, dim=1)
    tau = topk_vals[:, -1].unsqueeze(1)
    
    mask = (mag >= tau).float()
    delta_flat = delta_flat * mask
    return delta_flat.view(delta.size())

def sparse_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1):
    """
    Sparse-PGD baseline: PGD with exact L0 projection on features (coordinates).
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    B, C, H, W = images.shape
    N = C * H * W
    k_max = int(N * k_ratio)
    if k_max < 1: k_max = 1
    
    # Initialize with random noise in Linf ball
    delta = torch.empty_like(images).uniform_(-eps, eps)
    delta = project_l0_coordinates(delta, k_max)
    
    for t in range(iters):
        adv_images = images + delta
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, min=-eps, max=eps)
            delta = project_l0_coordinates(delta, k_max)
            
            # Box constraint
            adv_images = torch.clamp(images + delta, min=0, max=1)
            delta = adv_images - images
            delta = project_l0_coordinates(delta, k_max) # Re-project after clamp
            
    return images + delta
