import sys
import os

# Set cache and temp directories to be inside the workspace to comply with sandbox mounts
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

tmp_dir = os.path.join(workspace_dir, '.tmp')
os.makedirs(tmp_dir, exist_ok=True)
os.environ['TMPDIR'] = tmp_dir
os.environ['TEMP'] = tmp_dir
os.environ['TMP'] = tmp_dir

import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(workspace_dir)

from src.datasets.loader import get_cifar10, get_tiny_imagenet
from src.models.loader import get_model
from src.attacks.fgsm import fgsm_attack
from src.attacks.bim import bim_attack
from src.attacks.pgd import pgd_attack
from src.attacks.topk_pgd import topk_pgd_attack
from src.attacks.sparse_pgd import sparse_pgd_attack
from src.attacks.sparsefool import sparsefool_attack
from src.attacks.greedy_fool import greedy_fool_attack
from src.utils.metrics import get_metrics, calculate_psnr, calculate_ssim, calculate_lpips

def run_final_benchmark(num_batches=1, batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasets_list = ['cifar10']
    models_type = ['Standard', 'Robust']
    k_ratios = [0.1, 0.5, 1.0] # reduced for quick testing
    
    os.makedirs('results', exist_ok=True)
    csv_file = 'results/final_results.csv'
    cols = ['Dataset', 'Model', 'Attack', 'K-Ratio', 'Dynamic', 'Iteration', 'Accuracy', 'ASR', 'L0', 'Sparsity', 'L2', 'Linf', 'SSIM', 'PSNR', 'LPIPS', 'Time(s)']
    all_rows = []
    
    for dname in datasets_list:
        if dname == 'cifar10':
            loader = get_cifar10(batch_size=batch_size)
        else:
            loader = get_tiny_imagenet(batch_size=batch_size, resize=224)
            if loader is None: continue
            
        for mtype in models_type:
            print(f"\n--- Method: {dname} | Model: {mtype} ---")
            is_robust = (mtype == 'Robust')
            try:
                model = get_model('resnet18', dataset=dname, robust=is_robust).to(device)
            except Exception as e:
                print(f"Error loading model: {e}")
                continue
            model.eval()
            
            curr_num_batches = num_batches if num_batches else len(loader)
            
            for batch_idx, (images, labels) in enumerate(tqdm(loader, total=curr_num_batches)):
                if batch_idx >= curr_num_batches: break
                images, labels = images.to(device), labels.to(device)
                
                with torch.no_grad():
                    clean_outputs = model(images)
                    _, clean_preds = torch.max(clean_outputs, 1)
                    correct_idx = (clean_preds == labels)

                def log_metrics(adv_images, attack_name, k, dyn, step, duration=0):
                    with torch.no_grad():
                        adv_outputs = model(adv_images)
                        _, adv_preds = torch.max(adv_outputs, 1)
                        acc = (adv_preds == labels).float().mean().item()
                        asr = (adv_preds[correct_idx] != labels[correct_idx]).float().mean().item() if correct_idx.sum() > 0 else 0.0
                        diff = (adv_images - images).abs()
                        l0 = (diff.max(dim=1)[0] > 1e-4).float().view(diff.size(0), -1).sum(dim=1).mean().item()
                        sparsity = 1.0 - (l0 / (images.size(2)*images.size(3)))
                        diff_flat = diff.view(diff.size(0), -1)
                        l2 = torch.norm(diff_flat, p=2, dim=1).mean().item()
                        linf = torch.norm(diff_flat, p=float('inf'), dim=1).mean().item()
                        psnr = calculate_psnr(images, adv_images)
                        ssim = calculate_ssim(images, adv_images)
                        lpips = calculate_lpips(images, adv_images, device)
                        all_rows.append([dname, mtype, attack_name, k, dyn, step, acc, asr, l0, sparsity, l2, linf, ssim, psnr, lpips, duration])

                # 1. Clean
                start_time = time.time()
                log_metrics(images, 'Clean', 0, False, 0, time.time() - start_time)
                
                # 2. Dense Attacks
                start_time = time.time()
                adv_fgsm = fgsm_attack(model, images, labels, eps=8/255)
                log_metrics(adv_fgsm, 'FGSM', 0, False, 1, time.time() - start_time)
                
                start_time = time.time()
                _, history = pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, return_history=True)
                for t, adv_img in enumerate(history):
                    log_metrics(adv_img, 'PGD', 0, False, t+1, time.time() - start_time)
                
                # 3. Sparse SOTA Baselines
                start_time = time.time()
                adv_sparsefool = sparsefool_attack(model, images, labels, max_iters=20)
                log_metrics(adv_sparsefool, 'SparseFool', 0, False, 20, time.time() - start_time)

                for k in k_ratios:
                    start_time = time.time()
                    adv_sparse_pgd = sparse_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=k)
                    log_metrics(adv_sparse_pgd, 'Sparse-PGD', k, False, 10, time.time() - start_time)

                    start_time = time.time()
                    adv_greedy = greedy_fool_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=k)
                    log_metrics(adv_greedy, 'GreedyFool', k, False, 10, time.time() - start_time)

                    # 4. Our Proposed Sparse Attack (Top-k PGD)
                    start_time = time.time()
                    _, history = topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=k, dynamic=True, return_history=True)
                    for t, adv_img in enumerate(history):
                        log_metrics(adv_img, 'Proposed-TopkPGD', k, True, t+1, time.time() - start_time)
                
                if (batch_idx + 1) % 2 == 0:
                    pd.DataFrame(all_rows, columns=cols).to_csv(csv_file, index=False)
                    
    pd.DataFrame(all_rows, columns=cols).to_csv(csv_file, index=False)
    print(f"\nFinished. Results in {csv_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=1, help="Number of batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    args = parser.parse_args()
    
    run_final_benchmark(num_batches=args.batches, batch_size=args.batch_size)
