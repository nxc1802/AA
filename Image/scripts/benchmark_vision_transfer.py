import sys
import os
import torch
import pandas as pd
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import get_cifar10_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def main():
    output_dir = "Image/outputs/transfer"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading CIFAR-10...")
    dataloader = get_cifar10_loader(batch_size=1)
    
    # 1. Generate Patches using Source Model (ResNet-20)
    print("[*] Generating patches on cifar10_resnet20...")
    source_model = VisionModelWrapper(model_name="cifar10_resnet20")
    hessian_attack = SubspaceHessianAttack2D(source_model, patch_percent=0.1, num_iter=10, epsilon=0.031)
    
    adv_samples = []
    num_gen = 100
    
    count = 0
    for images, labels in tqdm(dataloader, desc="Generating Adv Samples", total=num_gen):
        if count >= num_gen:
            break
        
        # Only use samples correctly classified by source
        if source_model.predict(images).item() == labels.item():
            adv_images = hessian_attack.attack(images, labels)
            adv_samples.append((adv_images, labels.item()))
        
        count += 1
        
    # 2. Test Transfer to Target Models (ResNet-32, ResNet-44, ResNet-56)
    targets = ["cifar10_resnet32", "cifar10_resnet44", "cifar10_resnet56"]
    transfer_results = {}
    
    for target_name in targets:
        print(f"[*] Testing Transfer to {target_name}...")
        target_model = VisionModelWrapper(model_name=target_name)
        
        success_count = 0
        total_valid = len(adv_samples)
        
        for adv_img, label in tqdm(adv_samples, desc=f"Evaluating {target_name}"):
            pred = target_model.predict(adv_img).item()
            # If target model is fooled (pred != label), it's a transfer success
            if pred != label:
                success_count += 1
                
        tsr = (success_count / total_valid * 100) if total_valid > 0 else 0.0
        transfer_results[target_name] = tsr
        print(f"Done {target_name}: TSR = {tsr:.2f}%")
        
    with open(os.path.join(output_dir, "vision_transfer_summary.json"), "w") as f:
        json.dump(transfer_results, f, indent=4)
        
    print("\n" + "="*50)
    print("VISION TRANSFER BENCHMARK COMPLETE")
    print(f"Results saved to {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
