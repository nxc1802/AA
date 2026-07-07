import torch
import torch.nn as nn

def sparsefool_attack(model, images, labels, max_iters=20, lambda_val=3.0):
    """
    A simplified version of SparseFool algorithm (iterative coordinate-wise).
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    B, C, H, W = images.shape
    adv_images = images.clone()
    
    for b in range(B):
        x = images[b:b+1].clone()
        y = labels[b:b+1]
        
        x_adv = x.clone()
        for i in range(max_iters):
            x_adv.requires_grad = True
            out = model(x_adv)
            _, pred = torch.max(out, 1)
            if pred.item() != y.item():
                break
                
            cost = loss(out, y)
            grad = torch.autograd.grad(cost, x_adv, retain_graph=False, create_graph=False)[0]
            
            with torch.no_grad():
                grad_abs = grad.abs().view(1, -1)
                idx = torch.argmax(grad_abs, dim=1).item()
                
                val = grad.view(1, -1)[0, idx].sign()
                x_adv_flat = x_adv.view(1, -1)
                # Apply large perturbation to specific pixel coordinate
                x_adv_flat[0, idx] = torch.clamp(x_adv_flat[0, idx] + val * lambda_val, min=0, max=1)
                x_adv = x_adv_flat.view(1, C, H, W)
                
        adv_images[b:b+1] = x_adv.detach()
        
    return adv_images
