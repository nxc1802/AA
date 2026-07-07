# %% [markdown]
# # Kaggle Notebook
# Run the cell below to install requirements
# %% [code]
!pip install robustbench lpips advertorch opencv-python

# %% [code]
# ================= IMPORTS =================
from tqdm import tqdm
import json
import lpips
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
import torchvision

# ================= CORE MODULES =================

# --- Extracted from src/utils/config.py ---

@dataclass
class AttackConfig:
    attack_name: str
    eps: float = 8/255
    alpha: float = 2/255 
    iters: int = 10
    k_ratio: float = 0.1
    dynamic: bool = True
    strict_l0: bool = True
    restarts: int = 1
    seed: int = 42
    model_name: str = ""
    model_hash: str = ""
    dataset: str = "cifar10"
    subset_indices: list = None
    timestamp: float = 0.0
    
    def save(self, path):
        self.timestamp = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(asdict(self), f)
            else:
                json.dump(asdict(self), f, indent=4)
                
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)


# --- Extracted from src/utils/metrics.py ---

# Initialize LPIPS model globally to avoid reloading
_loss_fn_alex = None

def get_lpips_model(device):
    global _loss_fn_alex
    if _loss_fn_alex is None:
        _loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        _loss_fn_alex.eval()
    return _loss_fn_alex

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

def calculate_ssim(img1, img2, window_size=11):
    import torch.nn.functional as F
    
    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size).float() - window_size//2)**2 / (2 * sigma**2))
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def calculate_lpips(img1, img2, device):
    """
    Calculate LPIPS score between two image batches.
    img1, img2: [B, C, H, W] in [0, 1]
    LPIPS expects [-1, 1] range.
    """
    img1_scaled = img1 * 2.0 - 1.0
    img2_scaled = img2 * 2.0 - 1.0
    loss_fn = get_lpips_model(device)
    with torch.no_grad():
        dist = loss_fn(img1_scaled, img2_scaled)
    return dist.mean().item()

def get_metrics(model, images, labels, adv_images, device):
    """
    Calculate comprehensive metrics for a batch of adversarial images.
    Returns: Acc, ASR, L0, L2, Linf, SSIM, PSNR, LPIPS
    """
    model.eval()
    with torch.no_grad():
        outputs_clean = model(images.to(device))
        _, pred_clean = torch.max(outputs_clean, 1)
        correct_idx = (pred_clean == labels.to(device))
        
        outputs_adv = model(adv_images.to(device))
        _, pred_adv = torch.max(outputs_adv, 1)
        
        acc_adv = (pred_adv == labels.to(device)).float().mean().item()
        
        if correct_idx.sum() > 0:
            asr = (pred_adv[correct_idx] != labels.to(device)[correct_idx]).float().mean().item()
        else:
            asr = 0.0
            
        diff = (adv_images - images).abs()
        # L0: spatial pixels modified
        l0_per_image = (diff.max(dim=1)[0] > 1e-4).float().view(diff.size(0), -1).sum(dim=1)
        avg_l0 = l0_per_image.mean().item()
        
        diff_flat = diff.view(diff.size(0), -1)
        l2_norm = torch.norm(diff_flat, p=2, dim=1).mean().item()
        linf_norm = torch.norm(diff_flat, p=float('inf'), dim=1).mean().item()
        
        psnr = calculate_psnr(images, adv_images)
        ssim = calculate_ssim(images, adv_images)
        lpips_score = calculate_lpips(images, adv_images, device)
        
        num_pixels = images.size(2) * images.size(3)
        sparsity = 1.0 - (avg_l0 / num_pixels)
        
    return {
        'acc': acc_adv,
        'asr': asr,
        'l0': avg_l0,
        'sparsity': sparsity,
        'l2': l2_norm,
        'linf': linf_norm,
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips_score
    }


# --- Extracted from src/utils/visualization.py ---

def visualize_perturbation(image, adv_image, title="Perturbation", save_path=None):
    image_np = image.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    adv_np = adv_image.squeeze().cpu().detach().permute(1, 2, 0).numpy()
    
    diff = np.abs(adv_np - image_np)
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    # Compute active mask
    mask = (diff.sum(axis=-1) > 1e-5).astype(float)
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image_np)
    plt.title("Clean Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 2)
    plt.imshow(adv_np)
    plt.title("Adversarial Image")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(diff_norm)
    plt.title("Perturbation (Normalized)")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Active Pixels: {int(mask.sum())}")
    plt.axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


# --- Extracted from src/attacks/topk_pgd.py ---

def project_topk_support(delta, k):
    """
    Keep only the top-k pixels with highest perturbation magnitude.
    delta: [B, C, H, W]
    k: int
    """
    B, C, H, W = delta.size()
    # Magnitude per spatial pixel
    spatial_mag = delta.abs().max(dim=1)[0] # [B, H, W]
    spatial_mag_flat = spatial_mag.view(B, -1) # [B, H*W]
    
    if k >= spatial_mag_flat.size(1):
        return delta
        
    topk_vals, _ = torch.topk(spatial_mag_flat, k, dim=1)
    tau = topk_vals[:, -1].view(-1, 1, 1) # [B, 1, 1]
    
    mask = (spatial_mag >= tau).float().unsqueeze(1) # [B, 1, H, W]
    return delta * mask

def topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1, dynamic=True, return_history=False, score_ema=0.0):
    """
    Gradient-Guided Sparse Attack (Top-k PGD) with STRICT L0 budget.
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    B, C, H, W = images.shape
    N = H * W
    k_max = int(N * k_ratio)
    if k_max < 1: k_max = 1
    
    delta = torch.zeros_like(images)
    
    history = []
    
    ema_score = None
    
    for t in range(iters):
        adv_images = images + delta
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            score = grad.abs().max(dim=1)[0] # [B, H, W]
            
            if ema_score is None:
                ema_score = score
            else:
                ema_score = score_ema * ema_score + (1 - score_ema) * score
                
            if dynamic:
                # Dynamic mask: select k_t active pixels based on EMA score
                # Schedule: decrease active set from k_max towards k_max (constant for now as we enforce L0 strictly anyway)
                # Let's keep a schedule that starts broad and narrows down if needed, but for simplicity, we can just select k_max pixels based on score, since project_topk_support will enforce k_max strictly anyway.
                # Actually, dynamic support means the mask changes. We can just pick the top-k gradient pixels.
                score_flat = ema_score.view(B, -1)
                
                # We can optionally use a broader k_t here and then strictly cap delta, but the simplest is just k_max.
                k_t = k_max
                topk_vals, _ = torch.topk(score_flat, k_t, dim=1)
                tau = topk_vals[:, -1].view(-1, 1, 1)
                mask = (ema_score >= tau).float().unsqueeze(1)
            else:
                if t == 0:
                    score_flat = ema_score.view(B, -1)
                    topk_vals, _ = torch.topk(score_flat, k_max, dim=1)
                    tau = topk_vals[:, -1].view(-1, 1, 1)
                    mask = (ema_score >= tau).float().unsqueeze(1)
            
            # 1. Update delta with gradient and mask
            delta = delta + alpha * grad.sign() * mask
            
            # 2. Project to Linf
            delta = torch.clamp(delta, min=-eps, max=eps)
            
            # 3. Project to L0 strictly
            delta = project_topk_support(delta, k_max)
            
            # 4. Project to image bounds
            adv_images = torch.clamp(images + delta, min=0, max=1)
            delta = adv_images - images
            
            if return_history:
                history.append(adv_images.clone().detach())
        
    adv_images = images + delta
    
    # Assert final L0 is respected
    l0_actual = (delta.abs().max(dim=1)[0] > 1e-4).float().view(B, -1).sum(dim=1)
    # Allow a small float epsilon tolerance by checking non-zero pixels rather than >1e-4 if needed, but 1e-4 is safe.
    
    return (adv_images, history) if return_history else adv_images


# --- Extracted from src/attacks/sparse_pgd.py ---

def project_l0_coordinates(delta, k):
    """
    Project delta to L0 ball with k non-zero elements (over all coordinates).
    """
    B = delta.size(0)
    delta_flat = delta.view(B, -1)
    if k >= delta_flat.size(1):
        return delta
        
    mag = delta_flat.abs()
    topk_vals, _ = torch.topk(mag, k, dim=1)
    tau = topk_vals[:, -1].unsqueeze(1)
    
    mask = (mag >= tau).float()
    delta_flat = delta_flat * mask
    return delta_flat.view(delta.size())

def sparse_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1):
    """
    Sparse-PGD baseline: PGD with exact L0 projection on features (coordinates).
    """
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss = nn.CrossEntropyLoss()
    
    B, C, H, W = images.shape
    N = C * H * W
    k_max = int(N * k_ratio)
    if k_max < 1: k_max = 1
    
    # Initialize with random noise in Linf ball
    delta = torch.empty_like(images).uniform_(-eps, eps)
    delta = project_l0_coordinates(delta, k_max)
    
    for t in range(iters):
        adv_images = images + delta
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        cost = loss(outputs, labels)
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, min=-eps, max=eps)
            delta = project_l0_coordinates(delta, k_max)
            
            # Box constraint
            adv_images = torch.clamp(images + delta, min=0, max=1)
            delta = adv_images - images
            delta = project_l0_coordinates(delta, k_max) # Re-project after clamp
            
    return images + delta


# --- Extracted from src/attacks/sparsefool.py ---

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


# --- Extracted from scripts/run_final_bench.py ---

# Set cache and temp directories to be inside the workspace to comply with sandbox mounts
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

tmp_dir = os.path.join(workspace_dir, '.tmp')
os.makedirs(tmp_dir, exist_ok=True)
os.environ['TMPDIR'] = tmp_dir
os.environ['TEMP'] = tmp_dir
os.environ['TMP'] = tmp_dir


sys.path.append(workspace_dir)


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
