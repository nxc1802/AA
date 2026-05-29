import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, eps=8/255):
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    images.requires_grad = True
    outputs = model(images)
    
    cost = loss(outputs, labels)
    grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
    
    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    return adv_images
