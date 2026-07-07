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


# --- Extracted from scripts/run_defense_bench.py ---
warnings.filterwarnings("ignore", category=FutureWarning)

# Set cache and temp directories to be inside the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

sys.path.append(workspace_dir)

    MedianSmoothingDefense,
    BitReductionsDefense,
    JPEGCompressionDefense,
    RandomNoiseDefense,
    RandomizedSmoothingModel,
    FeatureDenoisingWrapper
)

def evaluate_defense(model, defense_name, defense_fn, images, labels, correct_idx):
    """
    Evaluates a model under a specific defense strategy on given batch of images.
    """
    if defense_name == "Randomized Smoothing (std=0.12, N=100)":
        smoothed_model = RandomizedSmoothingModel(model, sigma=0.12, N=100)
        outputs = smoothed_model(images)
    elif defense_name == "Feature Denoising (3x3 hooks)":
        denoised_model = FeatureDenoisingWrapper(model, kernel_size=3)
        outputs = denoised_model(images)
        denoised_model.remove_hooks()
    else:
        defended_images = defense_fn(images)
        outputs = model(defended_images)
        
    _, preds = torch.max(outputs, 1)
    acc = (preds == labels).float().mean().item()
    
    if correct_idx.sum() > 0:
        asr = (preds[correct_idx] != labels[correct_idx]).float().mean().item()
    else:
        asr = 0.0
        
    return acc, asr

def run_defense_benchmark(dataset='cifar10', num_batches=2, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"==================================================")
    print(f"Running Advanced Defense Evaluation from Pre-saved Attacks")
    print(f"Dataset: {dataset} | Device: {device}")
    print(f"==================================================")

    # 1. Models to evaluate
    models_dict = {
        "Standard ResNet-18": ('resnet18', False),
        "Robust ResNet-18 (AT)": ('resnet18', True),
        "TRADES robust model": ('trades', True),
        "GG-SAT ResNet-18 (GG-SAT)": ('gg_sat', True)
    }

    loaded_models = {}
    for name, (mname, is_robust) in models_dict.items():
        try:
            m = get_model(mname, dataset=dataset, robust=is_robust).to(device)
            m.eval()
            loaded_models[name] = m
        except Exception as e:
            print(f"Skipping {name} loading: {e}")

    if not loaded_models:
        print("Error: No models could be loaded.")
        return

    # 2. Check if pre-saved attacks exist. If not, generate them!
    adv_dir = os.path.join(workspace_dir, 'data', 'adv_images', dataset)
    
    # We will use "resnet18" (Standard) as the attack source for transfer robustness
    standard_adv_dir = os.path.join(adv_dir, 'resnet18')
    attacks_to_eval = ['Clean', 'FGSM', 'BIM', 'PGD', 'Sparse']
    
    missing_attacks = False
    for attack in attacks_to_eval:
        if not os.path.exists(os.path.join(standard_adv_dir, f"{attack}.pt")):
            missing_attacks = True
            break
            
    if missing_attacks:
        print("Pre-saved attacks not found or incomplete. Automatically generating attacks first...")
        generate_attacks(dataset=dataset, model_name='resnet18', num_batches=num_batches, batch_size=batch_size)
        
    # Also verify/generate direct attacks for robust models if needed
    for model_name, (mname, _) in models_dict.items():
        if model_name not in loaded_models: continue
        model_adv_dir = os.path.join(adv_dir, mname)
        if not os.path.exists(os.path.join(model_adv_dir, "PGD.pt")):
            print(f"Generating direct attacks for model {mname}...")
            generate_attacks(dataset=dataset, model_name=mname, num_batches=num_batches, batch_size=batch_size)

    # 3. Instantiate Defenses
    defenses = {
        "No Defense": lambda x: x,
        "Median Filter (3x3)": MedianSmoothingDefense(kernel_size=3).to(device),
        "Bit Reduction (3-bit)": BitReductionsDefense(bits=3).to(device),
        "JPEG Compression (Q75)": JPEGCompressionDefense(quality=75).to(device),
        "Random Noise (std=0.02)": RandomNoiseDefense(std=0.02).to(device),
        "Randomized Smoothing (std=0.12, N=100)": None,
        "Feature Denoising (3x3 hooks)": None
    }

    results_rows = []

    # 4. Evaluate each model
    for name, model in loaded_models.items():
        print(f"\nEvaluating defenses on model: {name}")
        mname = 'resnet18'
        if 'trades' in name.lower():
            mname = 'trades'
        elif 'gg-sat' in name.lower() or 'gg_sat' in name.lower():
            mname = 'gg_sat'
        elif 'resnet50' in name.lower():
            mname = 'resnet50'

        # Load Attacks
        try:
            # Direct PGD & Direct Sparse generated on current model
            direct_pgd_dict = torch.load(os.path.join(adv_dir, mname, "PGD.pt"), map_location=device)
            direct_sparse_dict = torch.load(os.path.join(adv_dir, mname, "Sparse.pt"), map_location=device)
            
            # Transfer attacks generated on standard model
            transfer_pgd_dict = torch.load(os.path.join(adv_dir, "resnet18", "PGD.pt"), map_location=device)
            transfer_sparse_dict = torch.load(os.path.join(adv_dir, "resnet18", "Sparse.pt"), map_location=device)
            
            # Clean
            clean_dict = torch.load(os.path.join(adv_dir, "resnet18", "Clean.pt"), map_location=device)
        except Exception as e:
            print(f"Failed to load attacks for model {name}: {e}. Skipping benchmark.")
            continue

        images = clean_dict['clean_images'].detach().to(device)
        labels = clean_dict['labels'].to(device)
        
        # Calculate clean indices
        with torch.no_grad():
            clean_outputs = model(images)
            _, clean_preds = torch.max(clean_outputs, 1)
            correct_idx = (clean_preds == labels)

        # Apply each defense
        with torch.no_grad():
            for def_name, defense_fn in defenses.items():
                acc_clean, _ = evaluate_defense(model, def_name, defense_fn, images, labels, correct_idx)
                
                acc_pgd_d, asr_pgd_d = evaluate_defense(model, def_name, defense_fn, direct_pgd_dict['adv_images'].detach().to(device), labels, correct_idx)
                acc_sparse_d, asr_sparse_d = evaluate_defense(model, def_name, defense_fn, direct_sparse_dict['adv_images'].detach().to(device), labels, correct_idx)
                
                acc_pgd_t, asr_pgd_t = evaluate_defense(model, def_name, defense_fn, transfer_pgd_dict['adv_images'].detach().to(device), labels, correct_idx)
                acc_sparse_t, asr_sparse_t = evaluate_defense(model, def_name, defense_fn, transfer_sparse_dict['adv_images'].detach().to(device), labels, correct_idx)
                
                results_rows.append({
                    "Model": name,
                    "Defense": def_name,
                    "Clean Acc": acc_clean,
                    "PGD Direct Acc": acc_pgd_d,
                    "PGD Direct ASR": asr_pgd_d,
                    "Sparse Direct Acc": acc_sparse_d,
                    "Sparse Direct ASR": asr_sparse_d,
                    "PGD Transfer Acc": acc_pgd_t,
                    "PGD Transfer ASR": asr_pgd_t,
                    "Sparse Transfer Acc": acc_sparse_t,
                    "Sparse Transfer ASR": asr_sparse_t
                })

    # 5. Save results to dataset-specific results folders
    dataset_res_dir = os.path.join(workspace_dir, 'results', dataset)
    os.makedirs(dataset_res_dir, exist_ok=True)
    
    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(dataset_res_dir, 'defense_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed CSV results saved to {csv_path}")

    report_path = os.path.join(dataset_res_dir, 'defense_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# Comprehensive Robustness Evaluation: {dataset.upper()} Defenses vs. Sparse Attacks\n\n")
        f.write("This report evaluates the effectiveness of standard preprocessing, certified, and feature-space defenses using **pre-saved decoupled adversarial images**.\n\n")
        
        f.write("## 1. Experimental Setup\n")
        f.write(f"- **Dataset**: {dataset}\n")
        f.write("- **Methodology**: Decoupled attack generation and defense evaluation.\n")
        f.write(f"- **Total Samples Evaluated**: {len(images)} samples.\n\n")
        
        f.write("## 2. Evaluation Results\n\n")
        
        for model_name in df["Model"].unique():
            f.write(f"### Results on {model_name}\n\n")
            df_model = df[df["Model"] == model_name].drop(columns=["Model"])
            
            df_formatted = df_model.copy()
            percent_cols = ["Clean Acc", "PGD Direct Acc", "PGD Direct ASR", "Sparse Direct Acc", "Sparse Direct ASR",
                            "PGD Transfer Acc", "PGD Transfer ASR", "Sparse Transfer Acc", "Sparse Transfer ASR"]
            for col in percent_cols:
                df_formatted[col] = (df_formatted[col] * 100).round(2).astype(str) + "%"
                
            headers = list(df_formatted.columns)
            markdown_table = "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for _, row in df_formatted.iterrows():
                markdown_table += "| " + " | ".join([str(val) for val in row]) + " |\n"
                
            f.write(markdown_table)
            f.write("\n\n")
            
    print(f"Evaluation report successfully saved to: {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tiny_imagenet", "imagenet"],
                        help="Dataset to run benchmark on")
    parser.add_argument("--batches", type=int, default=2, help="Number of batches for fallback generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    run_defense_benchmark(dataset=args.dataset, num_batches=args.batches, batch_size=args.batch_size)
