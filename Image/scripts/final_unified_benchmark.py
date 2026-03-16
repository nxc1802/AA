import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import get_cifar10_loader
from src.data.tiny_loader import get_tiny_imagenet_loader
from src.models.vision_wrapper import VisionModelWrapper
from src.attacks.baselines_2d import pgd_attack_2d
from src.attacks.pgd_patch_2d import PGDPatchAttack2D
from src.attacks.hessian_patch_2d import SubspaceHessianAttack2D

def denormalize(tensor, is_cifar=True):
    if is_cifar:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(tensor.device)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def get_metrics(orig, adv, is_cifar=True):
    orig_np = denormalize(orig, is_cifar).squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = denormalize(adv, is_cifar).squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    try:
        s = ssim(orig_np, adv_np, channel_axis=2, data_range=1.0)
    except:
        s = ssim(orig_np, adv_np, multichannel=True, data_range=1.0)
    l2 = np.linalg.norm((orig_np - adv_np).flatten())
    return s, l2

def run_benchmark(model_name, dataloader, num_samples=100, is_cifar=True):
    print(f"\n[>>>] Benchmarking: {model_name} ({'CIFAR' if is_cifar else 'Tiny-IN'})")
    wrapper = VisionModelWrapper(model_name=model_name)
    epsilon = 0.031
    num_iter = 10
    
    # Attacks
    pgd_global = lambda m, i, l: pgd_attack_2d(m, i, l, epsilon=epsilon, num_iter=num_iter)
    hess_patch = SubspaceHessianAttack2D(wrapper, patch_percent=0.1, num_iter=num_iter, epsilon=epsilon)
    
    stats = {
        "pgd_asr": 0, "hess_asr": 0,
        "pgd_ssim": [], "pgd_l2": [],
        "hess_ssim": [], "hess_l2": [],
        "total": 0
    }
    
    count = 0
    for images, labels in tqdm(dataloader, desc=model_name):
        if count >= num_samples: break
        
        images = images.to(wrapper.device)
        orig_pred = wrapper.predict(images)
        
        # PGD Global
        adv_pgd = pgd_global(wrapper, images, orig_pred)
        if wrapper.predict(adv_pgd) != orig_pred:
            stats["pgd_asr"] += 1
        s, l2 = get_metrics(images, adv_pgd, is_cifar)
        stats["pgd_ssim"].append(s)
        stats["pgd_l2"].append(l2)
        
        # Hessian Patch
        adv_hess = hess_patch.attack(images, orig_pred)
        if wrapper.predict(adv_hess) != orig_pred:
            stats["hess_asr"] += 1
        s, l2 = get_metrics(images, adv_hess, is_cifar)
        stats["hess_ssim"].append(s)
        stats["hess_l2"].append(l2)
        
        stats["total"] += 1
        count += 1
        
    return {
        "model": model_name,
        "pgd_asr": (stats["pgd_asr"]/stats["total"])*100,
        "pgd_ssim": np.mean(stats["pgd_ssim"]),
        "pgd_l2": np.mean(stats["pgd_l2"]),
        "hess_asr": (stats["hess_asr"]/stats["total"])*100,
        "hess_ssim": np.mean(stats["hess_ssim"]),
        "hess_l2": np.mean(stats["hess_l2"]),
    }

def main():
    cifar_models = ["cifar10_resnet20", "cifar10_resnet32", "cifar10_resnet44", "cifar10_resnet56"]
    tiny_models = ["resnet50", "resnet101", "resnet152"]
    
    results = []
    
    # 1. CIFAR Bench
    cifar_loader = get_cifar10_loader(batch_size=1)
    for m in cifar_models:
        results.append(run_benchmark(m, cifar_loader, is_cifar=True))
        
    # 2. Tiny-IN Bench
    tiny_loader = get_tiny_imagenet_loader(batch_size=1)
    for m in tiny_models:
        results.append(run_benchmark(m, tiny_loader, is_cifar=False))
        
    df = pd.DataFrame(results)
    output_path = "Image/outputs/final_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'PGD ASR':<8} | {'PGD SSIM':<8} | {'Hess ASR':<8} | {'Hess SSIM':<8} | {'L2 Ratio'}")
    print("-" * 80)
    for r in results:
        l2_ratio = r['hess_l2'] / r['pgd_l2']
        print(f"{r['model']:<20} | {r['pgd_asr']:<8.1f} | {r['pgd_ssim']:<8.3f} | {r['hess_asr']:<8.1f} | {r['hess_ssim']:<8.3f} | {l2_ratio:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
