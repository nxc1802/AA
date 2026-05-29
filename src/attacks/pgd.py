import torch
import torch.nn as nn

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, return_history=False):
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    history = []
    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        if return_history:
            history.append(adv_images.clone().detach())
        
    return (adv_images, history) if return_history else adv_images
