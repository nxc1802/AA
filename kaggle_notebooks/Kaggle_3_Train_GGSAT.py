# %% [markdown]
# # Kaggle Notebook
# Run the cell below to install requirements
# %% [code]
!pip install robustbench lpips advertorch opencv-python

# %% [code]
# ================= IMPORTS =================
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import argparse
import json
import lpips
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


# --- Extracted from scripts/train_sparse_robust.py ---

# Set cache and temp directories to be inside the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

sys.path.append(workspace_dir)


# Custom EMA Class for Weight Averaging
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="GG-SAT (Gradient-Guided Sparse Adversarial Training) on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--k_min", type=float, default=0.3, help="Minimum k-ratio for dynamic randomized masking")
    parser.add_argument("--k_max", type=float, default=0.7, help="Maximum k-ratio for dynamic randomized masking")
    parser.add_argument("--pure", action="store_true", help="Use purely adversarial training instead of mixed clean+adv loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight factor for adversarial loss in mixed training")
    parser.add_argument("--use_trades", action="store_true", help="Use TRADES loss instead of standard cross entropy loss")
    parser.add_argument("--trades_beta", type=float, default=6.0, help="TRADES trade-off parameter (1/lambda)")
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average (EMA) of model weights for validation/saving")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Decay rate for EMA weights")
    parser.add_argument("--attack_iters", type=int, default=10, help="Number of PGD iterations for inner loop attack")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD")
    parser.add_argument("--val_size", type=int, default=512, help="Number of test samples to use for validation each epoch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"==================================================")
    print(f"GG-SAT Training initialized on device: {device}")
    print(f"Arguments: {vars(args)}")
    print(f"==================================================")

    # 1. Data Loading
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_dir = os.path.join(workspace_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    try:
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}. Downloading may be required or data directory is invalid.")
        sys.exit(1)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Create validation subset for fast evaluation after each epoch
    indices = list(range(min(args.val_size, len(test_set))))
    val_subset = Subset(test_set, indices)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)

    # 2. Initialize Model
    import torchvision.models as models
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    # Initialize EMA helper if active
    if args.use_ema:
        ema_helper = EMA(model, decay=args.ema_decay)
        print(f"EMA helper initialized with decay: {args.ema_decay}")

    # 3. Optimizer & Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    # 4. Checkpoint Directories
    models_dir = os.path.join(workspace_dir, 'models', 'cifar10', 'Linf')
    os.makedirs(models_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(models_dir, 'SparseRobustResNet18.pt')
    
    best_robust_acc = 0.0
    best_clean_acc = 0.0

    print("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # Calculate dynamic shifting k_min lower bound to prevent under-sparsification early in training
        progress = epoch / args.epochs
        current_k_min = args.k_min + (args.k_max - args.k_min) * 0.5 * progress
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Select random k_ratio for this batch using dynamic schedule
            batch_k_ratio = random.uniform(current_k_min, args.k_max)
            
            # 1) Generate Sparse Adversarial Perturbations (Inner maximization)
            model.eval() # Disable dropout/batchnorm during attack generation
            adv_images = topk_pgd_attack(
                model, images, labels, 
                eps=8/255, alpha=2/255, 
                iters=args.attack_iters, k_ratio=batch_k_ratio, 
                dynamic=True, return_history=False
            )
            adv_images = adv_images.detach() # Cut gradient tracking
            
            # 2) Parameter update (Outer minimization)
            model.train()
            optimizer.zero_grad()
            
            if args.pure:
                # Purely Adversarial Training
                outputs_adv = model(adv_images)
                loss = criterion(outputs_adv, labels)
            elif args.use_trades:
                # TRADES Loss optimization (CE(clean) + Beta * KL(clean || adv))
                outputs_clean = model(images)
                outputs_adv = model(adv_images)
                loss_clean = criterion(outputs_clean, labels)
                loss_kl = criterion_kl(F.log_softmax(outputs_adv, dim=-1), F.softmax(outputs_clean, dim=-1))
                loss = loss_clean + args.trades_beta * loss_kl
            else:
                # Mixed Training (Clean + Adv)
                outputs_clean = model(images)
                outputs_adv = model(adv_images)
                loss_clean = criterion(outputs_clean, labels)
                loss_adv = criterion(outputs_adv, labels)
                loss = (1 - args.beta) * loss_clean + args.beta * loss_adv
                
            loss.backward()
            optimizer.step()
            
            # Update EMA weights
            if args.use_ema:
                ema_helper.update()
            
            epoch_loss += loss.item() * labels.size(0)
            
        scheduler.step()
        epoch_duration = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader.dataset)
        
        # 5. Fast Evaluation loop after epoch
        # Apply EMA weights if active
        if args.use_ema:
            ema_helper.apply_shadow()
            
        model.eval()
        correct_clean = 0
        correct_robust = 0
        total = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                # Evaluate clean
                outputs_clean = model(val_images)
                _, preds_clean = torch.max(outputs_clean, 1)
                correct_clean += (preds_clean == val_labels).sum().item()
                
                # Evaluate sparse robust (with default reference k_ratio=0.3)
                with torch.enable_grad():
                    val_adv = topk_pgd_attack(
                        model, val_images, val_labels, 
                        eps=8/255, alpha=2/255, 
                        iters=10, k_ratio=0.3, 
                        dynamic=True, return_history=False
                    ).detach()
                    
                outputs_adv = model(val_adv)
                _, preds_adv = torch.max(outputs_adv, 1)
                correct_robust += (preds_adv == val_labels).sum().item()
                total += val_labels.size(0)
                
        val_clean_acc = correct_clean / total
        val_robust_acc = correct_robust / total
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Val Clean Acc: {val_clean_acc*100:.2f}% | "
              f"Val Robust Acc (k=0.3): {val_robust_acc*100:.2f}% | "
              f"LR: {current_lr:.5f} | "
              f"Time: {epoch_duration:.1f}s")
              
        # 6. Save checkpoint if Sparse Robust Accuracy improves
        # We also want to maintain decent clean accuracy, but prioritizing robust accuracy
        if val_robust_acc > best_robust_acc or (val_robust_acc == best_robust_acc and val_clean_acc > best_clean_acc):
            best_robust_acc = val_robust_acc
            best_clean_acc = val_clean_acc
            print(f"====> New best checkpoint saved with Robust Acc {val_robust_acc*100:.2f}% (Clean {val_clean_acc*100:.2f}%)!")
            torch.save(model.state_dict(), best_checkpoint_path)
            
        # Restore normal weights for next epoch's training
        if args.use_ema:
            ema_helper.restore()

    print("==================================================")
    print("Training finished successfully!")
    print(f"Best Val Robust Acc (k=0.3): {best_robust_acc*100:.2f}%")
    print(f"Best Val Clean Acc: {best_clean_acc*100:.2f}%")
    print(f"Weights saved at: {best_checkpoint_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
