# %% [markdown]
# # Kaggle Notebook: GG-SAT (Gradient-Guided Sparse Adversarial Training)
# This notebook is fully self-contained and runs on Kaggle (with GPU accelerated environment).
# It implements the custom adversarial training method **GG-SAT** from scratch on CIFAR-10,
# incorporating advanced enhancements:
# 1. CIFAR-adapted ResNet-18 architecture (3x3 conv1, no maxpool).
# 2. Inner-loop Top-K PGD attack with dynamic randomized masking and custom iterations.
# 3. Dynamic $k$-ratio scheduling (lower-bound shifting as training progresses).
# 4. Optional TRADES Loss optimization (Kullback-Leibler divergence constraints).
# 5. Optional Exponential Moving Average (EMA) weight averaging.

# %% [code]
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
import numpy as np
from torch.utils.data import DataLoader, Subset

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [code]
# 1. Define CIFAR-adapted ResNet-18
def make_cifar_resnet18(num_classes=10):
    model = tv_models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# %% [code]
# 2. Define Custom EMA Class for Weight Averaging
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

# %% [code]
# 3. Define Top-K PGD Attack for GG-SAT Inner Loop
def topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1, dynamic=True):
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss_fn = nn.CrossEntropyLoss()
    
    adv_images = images.clone().detach()
    mask = None
    
    for t in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            if dynamic or (t == 0):
                score = grad.abs()
                score_flatten = score.view(score.size(0), -1)
                N = score_flatten.size(1)
                
                k_max = int(N * k_ratio)
                if dynamic:
                    k_t = int(k_max * (1 - t / iters))
                else:
                    k_t = k_max
                if k_t < 1: k_t = 1
                
                topk_vals, _ = torch.topk(score_flatten, k_t, dim=1)
                tau = topk_vals[:, -1].view(-1, 1, 1, 1)
                mask = (score >= tau).float()
                
            update = alpha * grad.sign() * mask
            adv_images.data.copy_(images + torch.clamp(adv_images + update - images, min=-eps, max=eps))
            adv_images.data.clamp_(0, 1)
            
    return adv_images

# %% [code]
# 4. Load CIFAR-10 Data with Augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
])

print("Downloading CIFAR-10 dataset...")
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

# Create a small validation loader for fast evaluation during training
val_subset = Subset(test_set, list(range(512)))
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)

# %% [code]
# 5. GG-SAT Training Parameters & Configuration
epochs = 5  # Set to 5 for fast demo in Kaggle. Set to 50 or 100 for production.
lr = 0.1
weight_decay = 5e-4

# GG-SAT configuration
k_min = 0.3
k_max = 0.7
attack_iters = 10   # Configurable inner loop attack steps (recommended: 10)
beta = 0.5          # Weight factor for mixed training loss (used when pure=False and use_trades=False)
pure = False        # Purely Adversarial Training (only uses adv loss)

# TRADES & EMA Upgrades
use_trades = True   # Enable TRADES KL loss instead of standard mixed loss
trades_beta = 6.0   # TRADES trade-off parameter (1/lambda)
use_ema = True      # Enable Exponential Moving Average weight averaging
ema_decay = 0.999

# Initialize Model, Optimizer, Scheduler and Loss
model = make_cifar_resnet18().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()
criterion_kl = nn.KLDivLoss(reduction='batchmean')

# Initialize EMA helper if enabled
if use_ema:
    ema_helper = EMA(model, decay=ema_decay)
    print(f"EMA weight averaging enabled with decay: {ema_decay}")

best_robust_acc = 0.0
best_checkpoint_path = 'SparseRobustResNet18.pt'

print(f"\nInitialized GG-SAT training on device: {device}")
print(f"Epochs: {epochs} | LR: {lr} | k_range: [{k_min}, {k_max}] | Inner-Attack Steps: {attack_iters}")
if use_trades:
    print(f"Loss Function: TRADES (beta={trades_beta})")
else:
    print(f"Loss Function: Mixed Cross-Entropy (beta={beta})")
print("="*60)

# %% [code]
# 6. GG-SAT Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    
    # Calculate dynamic lower bound for k to prevent under-sparsification early in training
    progress = epoch / epochs
    current_k_min = k_min + (k_max - k_min) * 0.5 * progress
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Pick dynamic randomized k-ratio for this batch
        batch_k_ratio = random.uniform(current_k_min, k_max)
        
        # 1) Generate Sparse Adversarial Perturbations (Inner maximization)
        model.eval()
        adv_images = topk_pgd_attack(
            model, images, labels, 
            eps=8/255, alpha=2/255, 
            iters=attack_iters, k_ratio=batch_k_ratio, 
            dynamic=True
        )
        adv_images = adv_images.detach() # Detach to stop gradient tracking of attack generation
        
        # 2) Parameter update (Outer minimization)
        model.train()
        optimizer.zero_grad()
        
        if pure:
            # Purely Adversarial Training
            outputs_adv = model(adv_images)
            loss = criterion(outputs_adv, labels)
        elif use_trades:
            # TRADES Loss optimization: CE(clean) + trades_beta * KL(clean || adv)
            outputs_clean = model(images)
            outputs_adv = model(adv_images)
            loss_clean = criterion(outputs_clean, labels)
            loss_kl = criterion_kl(F.log_softmax(outputs_adv, dim=-1), F.softmax(outputs_clean, dim=-1))
            loss = loss_clean + trades_beta * loss_kl
        else:
            # Mixed CE Training
            outputs_clean = model(images)
            outputs_adv = model(adv_images)
            loss_clean = criterion(outputs_clean, labels)
            loss_adv = criterion(outputs_adv, labels)
            loss = (1 - beta) * loss_clean + beta * loss_adv
            
        loss.backward()
        optimizer.step()
        
        # Update EMA weights
        if use_ema:
            ema_helper.update()
            
        epoch_loss += loss.item() * labels.size(0)
        
    scheduler.step()
    epoch_duration = time.time() - start_time
    avg_loss = epoch_loss / len(train_loader.dataset)
    
    # 3) Fast evaluation after epoch
    # Apply EMA weights if enabled
    if use_ema:
        ema_helper.apply_shadow()
        
    model.eval()
    correct_clean = 0
    correct_robust = 0
    total = 0
    
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            
            # Clean Evaluation
            outputs_clean = model(val_images)
            _, preds_clean = torch.max(outputs_clean, 1)
            correct_clean += (preds_clean == val_labels).sum().item()
            
            # Sparse Robust Evaluation (reference k=0.3, 10 iterations)
            with torch.enable_grad():
                val_adv = topk_pgd_attack(
                    model, val_images, val_labels, 
                    eps=8/255, alpha=2/255, 
                    iters=10, k_ratio=0.3, 
                    dynamic=True
                ).detach()
            outputs_adv = model(val_adv)
            _, preds_adv = torch.max(outputs_adv, 1)
            correct_robust += (preds_adv == val_labels).sum().item()
            total += val_labels.size(0)
            
    val_clean_acc = 100 * correct_clean / total
    val_robust_acc = 100 * correct_robust / total
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | "
          f"Val Clean: {val_clean_acc:.2f}% | Val Robust (k=0.3): {val_robust_acc:.2f}% | "
          f"LR: {current_lr:.5f} | Time: {epoch_duration:.1f}s")
          
    # 4) Save checkpoint if robust accuracy improves
    if val_robust_acc > best_robust_acc:
        best_robust_acc = val_robust_acc
        print(f"  ==> New best robust accuracy ({val_robust_acc:.2f}%)! Saving model checkpoint...")
        torch.save(model.state_dict(), best_checkpoint_path)
        
    # Restore normal weights for next epoch's training
    if use_ema:
        ema_helper.restore()

print("="*60)
print(f"Training completed. Best Robust Accuracy: {best_robust_acc:.2f}%")
print(f"Weights saved at: {best_checkpoint_path}")

# %% [code]
# 7. How to Load and Use
print("\nDemonstrating how to reload the trained model weights:")
reloaded_model = make_cifar_resnet18().to(device)
reloaded_model.load_state_dict(torch.load(best_checkpoint_path))
reloaded_model.eval()
print("Model loaded successfully from disk!")
