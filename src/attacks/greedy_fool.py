import torch
import torch.nn as nn

def greedy_fool_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1):
    """
    GreedyFool-like attack: uses gradient saliency with distortion awareness.
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    B, C, H, W = images.shape
    N = H * W
    k_max = int(N * k_ratio)
    if k_max < 1: k_max = 1
    
    delta = torch.zeros_like(images)
    
    for t in range(iters):
        adv_images = images + delta
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            # Greedy selection: combine gradient with distortion penalty
            # Saliency = |grad| / (1 + |delta|)
            saliency = grad.abs() / (1.0 + delta.abs())
            saliency_spatial = saliency.max(dim=1)[0]
            
            saliency_flat = saliency_spatial.view(B, -1)
            topk_vals, _ = torch.topk(saliency_flat, k_max, dim=1)
            tau = topk_vals[:, -1].view(-1, 1, 1)
            
            mask = (saliency_spatial >= tau).float().unsqueeze(1)
            
            delta = delta + alpha * grad.sign() * mask
            delta = torch.clamp(delta, min=-eps, max=eps)
            
            adv_images = torch.clamp(images + delta, min=0, max=1)
            delta = adv_images - images
            
    return images + delta
