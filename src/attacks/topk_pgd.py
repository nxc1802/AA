import torch
import torch.nn as nn

def topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1, dynamic=True, return_history=False):
    """
    Top-k PGD Attack with optional history recording.
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    adv_images = images.clone().detach() 
    
    mask = None
    history = []
    
    for t in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            if dynamic or (t == 0):
                score = grad.abs()
                score_flatten = score.view(score.size(0), -1)
                N = score_flatten.size(1)
                
                k_max = int(N * k_ratio)
                if dynamic:
                    k_t = int(k_max * (1 - t / iters))
                else:
                    k_t = k_max
                    
                if k_t < 1: k_t = 1
                
                topk_vals, _ = torch.topk(score_flatten, k_t, dim=1)
                tau = topk_vals[:, -1].view(-1, 1, 1, 1)
                mask = (score >= tau).float()

            update = alpha * grad.sign() * mask
            adv_images.data.copy_(images + torch.clamp(adv_images + update - images, min=-eps, max=eps))
            adv_images.data.clamp_(0, 1)
            
            if return_history:
                history.append(adv_images.clone().detach())
        
    return (adv_images, history) if return_history else adv_images
