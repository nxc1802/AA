# %% [markdown]
# # Kaggle Notebook: Standard Model Adversarial Attack Evaluation
# This notebook is fully self-contained and runs on Kaggle (with GPU accelerated environment).
# It trains a standard ResNet-18 classifier on CIFAR-10 and evaluates its robustness against:
# 1. Clean images
# 2. Dense FGSM Attack
# 3. Dense PGD Attack (10 iterations)
# 4. Sparse Top-K PGD Attack with Dynamic Masking (for various k-ratios: 0.1, 0.3, 0.5)

# %% [code]
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
import numpy as np
import pandas as pd
from tqdm import tqdm

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
# 2. Define Attack Algorithms
def fgsm_attack(model, images, labels, eps=8/255):
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss_fn = nn.CrossEntropyLoss()
    
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
    
    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    return adv_images

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10):
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    loss_fn = nn.CrossEntropyLoss()
    
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_images

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
# 3. Load CIFAR-10 Data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

print("Downloading and preparing CIFAR-10 dataset...")
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

# %% [code]
# 4. Train a Standard Model (Fast Training for demo, e.g. 5 epochs)
model = make_cifar_resnet18().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
criterion = nn.CrossEntropyLoss()

epochs = 5
print(f"Training standard ResNet-18 model for {epochs} epochs to establish baseline...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Time: {time.time() - start_time:.1f}s")

# %% [code]
# 5. Evaluate Clean Accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

clean_acc = 100 * correct / total
print(f"\nClean Test Accuracy of Standard ResNet-18: {clean_acc:.2f}%")

# %% [code]
# 6. Evaluate Robustness Against Attacks (on 1000 samples for speed)
eval_size = 1000
eval_images = []
eval_labels = []
samples_collected = 0
for img, lbl in test_loader:
    if samples_collected >= eval_size:
        break
    size_to_add = min(eval_size - samples_collected, img.size(0))
    eval_images.append(img[:size_to_add])
    eval_labels.append(lbl[:size_to_add])
    samples_collected += size_to_add

eval_images = torch.cat(eval_images, dim=0).to(device)
eval_labels = torch.cat(eval_labels, dim=0).to(device)

print(f"\nRunning adversarial evaluations on {eval_size} test samples...")

# A. FGSM
adv_fgsm = fgsm_attack(model, eval_images, eval_labels, eps=8/255)
with torch.no_grad():
    out_fgsm = model(adv_fgsm)
    _, pred_fgsm = torch.max(out_fgsm, 1)
    acc_fgsm = 100 * (pred_fgsm == eval_labels).float().mean().item()
print(f"FGSM Robust Accuracy (eps=8/255): {acc_fgsm:.2f}%")

# B. PGD-10
adv_pgd = pgd_attack(model, eval_images, eval_labels, eps=8/255, alpha=2/255, iters=10)
with torch.no_grad():
    out_pgd = model(adv_pgd)
    _, pred_pgd = torch.max(out_pgd, 1)
    acc_pgd = 100 * (pred_pgd == eval_labels).float().mean().item()
print(f"PGD-10 Robust Accuracy (eps=8/255, alpha=2/255): {acc_pgd:.2f}%")

# C. Sparse PGD (k = 0.1, 0.3, 0.5)
sparse_results = {}
for k in [0.1, 0.3, 0.5]:
    adv_sparse = topk_pgd_attack(model, eval_images, eval_labels, eps=8/255, alpha=2/255, iters=10, k_ratio=k, dynamic=True)
    with torch.no_grad():
        out_sparse = model(adv_sparse)
        _, pred_sparse = torch.max(out_sparse, 1)
        acc_sparse = 100 * (pred_sparse == eval_labels).float().mean().item()
    sparse_results[k] = acc_sparse
    print(f"Sparse PGD Robust Accuracy (k={k}): {acc_sparse:.2f}%")

# %% [code]
# 7. Print Summary Report
results_summary = [
    {"Attack": "Clean (No Attack)", "Robust Accuracy": f"{clean_acc:.2f}%", "ASR": "0.00%"},
    {"Attack": "FGSM (eps=8/255)", "Robust Accuracy": f"{acc_fgsm:.2f}%", "ASR": f"{clean_acc - acc_fgsm:.2f}%"},
    {"Attack": "PGD-10 (eps=8/255)", "Robust Accuracy": f"{acc_pgd:.2f}%", "ASR": f"{clean_acc - acc_pgd:.2f}%"},
    {"Attack": "Sparse PGD (k=0.1)", "Robust Accuracy": f"{sparse_results[0.1]:.2f}%", "ASR": f"{clean_acc - sparse_results[0.1]:.2f}%"},
    {"Attack": "Sparse PGD (k=0.3)", "Robust Accuracy": f"{sparse_results[0.3]:.2f}%", "ASR": f"{clean_acc - sparse_results[0.3]:.2f}%"},
    {"Attack": "Sparse PGD (k=0.5)", "Robust Accuracy": f"{sparse_results[0.5]:.2f}%", "ASR": f"{clean_acc - sparse_results[0.5]:.2f}%"},
]
df = pd.DataFrame(results_summary)
print("\n" + "="*50)
print(" STANDARD MODEL ADVERSARIAL ROBUSTNESS REPORT")
print("="*50)
print(df.to_string(index=False))
print("="*50)
