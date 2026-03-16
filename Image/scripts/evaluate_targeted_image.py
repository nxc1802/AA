import sys
import os
import torch
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import get_cifar10_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def main():
    output_dir = "Image/outputs/targeted"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading CIFAR-10 ResNet-20...")
    model_wrapper = VisionModelWrapper(model_name="cifar10_resnet20")
    dataloader = get_cifar10_loader(batch_size=1)
    
    # Setup Targeted Attack
    # We will target class 3 (cat) for all samples that are NOT cats
    target_class = 3
    hessian_targeted = SubspaceHessianAttack2D(model_wrapper, patch_percent=0.1, num_iter=10, epsilon=0.031, is_targeted=True)
    
    results = []
    num_samples = 100
    target_tensor = torch.tensor([target_class])
    
    count = 0
    for images, labels in tqdm(dataloader, desc="Targeted Attack (Goal: Cat)"):
        if count >= num_samples:
            break
            
        # Only attack if it's NOT already the target class and correctly classified
        orig_pred = model_wrapper.predict(images).item()
        if labels.item() != target_class and orig_pred == labels.item():
            adv_images = hessian_targeted.attack(images, labels, target_label=target_tensor)
            adv_pred = model_wrapper.predict(adv_images).item()
            
            success = (adv_pred == target_class)
            results.append({
                "id": count,
                "orig_label": labels.item(),
                "adv_label": adv_pred,
                "success": success
            })
            count += 1
            
    df_results = {
        "target_class": target_class,
        "total_samples": len(results),
        "success_rate": sum([r["success"] for r in results]) / len(results) * 100 if results else 0
    }
    
    with open(os.path.join(output_dir, "targeted_cat_results.json"), "w") as f:
        json.dump(df_results, f, indent=4)
        
    print("\n" + "="*50)
    print("TARGETED IMAGE ATTACK COMPLETE")
    print(f"Target: CAT | Success Rate: {df_results['success_rate']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
