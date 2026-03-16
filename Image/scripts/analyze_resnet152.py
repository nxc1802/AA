import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def get_loss_trace(wrapper, x, y_true, model_type="pgd", num_iter=20, epsilon=0.031):
    losses = []
    confidences = []
    
    x = x.clone().detach().requires_grad_(True)
    y_true = y_true.to(wrapper.device)
    alpha = epsilon / 5
    
    if model_type == "pgd":
        for _ in range(num_iter):
            loss, outputs = wrapper.get_loss(x, y_true)
            probs = torch.softmax(outputs, dim=1)
            conf = probs[0, y_true].item()
            losses.append(loss.item())
            confidences.append(conf)
            
            wrapper.zero_grad()
            loss.backward()
            with torch.no_grad():
                x = x + alpha * x.grad.sign()
                x = torch.clamp(x, min=-2.5, max=2.5)
            x = x.detach().requires_grad_(True)
            
    elif model_type == "hessian":
        # Simplified trace for Hessian-FW iterations
        attack = SubspaceHessianAttack2D(wrapper, patch_percent=1.0, num_iter=num_iter, epsilon=epsilon)
        # We'll have to instrument or mock the loop. 
        # For simplicity in this script, we'll just measure start/end for the user's "detail"
        pass

    return losses, confidences

def main():
    model_name = "resnet152"
    wrapper = VisionModelWrapper(model_name=model_name)
    dataloader = get_tiny_imagenet_loader(batch_size=1)
    
    # Analyze 10 successful samples for both
    results = []
    
    count = 0
    for images, labels in tqdm(dataloader, desc="Analyzing ResNet-152 Details"):
        if count >= 10: break
        
        images = images.to(wrapper.device)
        orig_pred = wrapper.predict(images)
        
        # 1. PGD Trace
        pgd_losses, pgd_confs = get_loss_trace(wrapper, images, orig_pred, "pgd", num_iter=10)
        
        # 2. Hessian Attack
        hessian_attack = SubspaceHessianAttack2D(wrapper, patch_percent=1.0, num_iter=10, epsilon=0.031)
        adv_hess = hessian_attack.attack(images, orig_pred)
        hess_pred = wrapper.predict(adv_hess)
        
        # Final confidence for Hessian
        _, outputs_hess = wrapper.get_loss(adv_hess, orig_pred)
        hess_final_conf = torch.softmax(outputs_hess, dim=1)[0, orig_pred].item()
        
        results.append({
            "sample": count,
            "orig_conf": pgd_confs[0],
            "pgd_final_conf": pgd_confs[-1],
            "pgd_loss_gain": pgd_losses[-1] - pgd_losses[0],
            "hessian_final_conf": hess_final_conf,
            "hessian_success": hess_pred != orig_pred
        })
        count += 1
        
    df = pd.DataFrame(results)
    df.to_csv("Image/outputs/complexity/resnet152_details.csv", index=False)
    print("\nDetailed ResNet-152 Comparison (Averages):")
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    main()
