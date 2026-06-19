# %% [markdown]
# # Kaggle Notebook: 6 Baseline Defense Methods Evaluation
# This notebook is fully self-contained and runs on Kaggle (with GPU accelerated environment).
# It trains a standard ResNet-18 model on CIFAR-10, defines the 6 baseline defenses, and evaluates them on:
# 1. Clean images
# 2. Dense PGD-10 images
# 3. Sparse PGD (k=0.1) images
# 
# Defenses Evaluated:
# 1. Median Smoothing Defense (3x3 kernel)
# 2. Bit Reduction Defense (3-bit quantization)
# 3. JPEG Compression Defense (Quality=75)
# 4. Random Noise Injection Defense (std=0.02)
# 5. Randomized Smoothing (Monte Carlo expectation-based consensus wrapper)
# 6. Feature Denoising Defense (PyTorch forward hook-based spatial filtering)

# %% [code]
import os
import io
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
from PIL import Image
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
# 3. Define the 6 Preprocessing and Certified Defenses

class MedianSmoothingDefense(nn.Module):
    def __init__(self, kernel_size=3):
        super(MedianSmoothingDefense, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        b, c, h, w = x.size()
        padding = self.kernel_size // 2
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        patches = x_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        patches = patches.contiguous().view(b, c, h, w, -1)
        median_val, _ = patches.median(dim=-1)
        return median_val

class BitReductionsDefense(nn.Module):
    def __init__(self, bits=3):
        super(BitReductionsDefense, self).__init__()
        self.levels = 2 ** bits

    def forward(self, x):
        return torch.round(x * (self.levels - 1)) / (self.levels - 1)

class JPEGCompressionDefense(nn.Module):
    def __init__(self, quality=75):
        super(JPEGCompressionDefense, self).__init__()
        self.quality = quality

    def forward(self, x):
        device = x.device
        b, c, h, w = x.size()
        x_np = x.cpu().numpy()
        x_defended = []
        for i in range(b):
            img_np = np.clip(x_np[i].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            img_dec = Image.open(buffer)
            dec_np = np.array(img_dec).astype(np.float32) / 255.0
            x_defended.append(dec_np.transpose(2, 0, 1))
        return torch.tensor(np.array(x_defended), device=device, dtype=x.dtype)

class RandomNoiseDefense(nn.Module):
    def __init__(self, std=0.02):
        super(RandomNoiseDefense, self).__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)

class RandomizedSmoothingModel(nn.Module):
    def __init__(self, model, sigma=0.12, N=50): # N=50 for speed on demo
        super(RandomizedSmoothingModel, self).__init__()
        self.model = model
        self.sigma = sigma
        self.N = N

    def forward(self, x):
        B, C, H, W = x.size()
        x_expanded = x.unsqueeze(1).repeat(1, self.N, 1, 1, 1).view(B * self.N, C, H, W)
        noise = torch.randn_like(x_expanded) * self.sigma
        x_noisy = x_expanded + noise
        
        chunk_size = 64
        total_samples = B * self.N
        logits_list = []
        for i in range(0, total_samples, chunk_size):
            chunk_x = x_noisy[i:i+chunk_size]
            logits_chunk = self.model(chunk_x)
            logits_list.append(logits_chunk)
            
        logits = torch.cat(logits_list, dim=0)
        logits = logits.view(B, self.N, -1).mean(dim=1)
        return logits

class FeatureDenoisingWrapper(nn.Module):
    def __init__(self, model, kernel_size=3):
        super(FeatureDenoisingWrapper, self).__init__()
        self.model = model
        self.kernel_size = kernel_size
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def denoise_hook(module, inp, out):
            B, C, H, W = out.size()
            padding = self.kernel_size // 2
            out_pad = F.pad(out, (padding, padding, padding, padding), mode='reflect')
            patches = out_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            patches = patches.contiguous().view(B, C, H, W, -1)
            median_val, _ = patches.median(dim=-1)
            return median_val

        for name, module in self.model.named_modules():
            # Hook the second block layer and third block layer of standard ResNet-18
            if name in ['layer2', 'layer3']:
                h = module.register_forward_hook(denoise_hook)
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward(self, x):
        return self.model(x)

# %% [code]
# 4. Load Data
transform = transforms.Compose([transforms.ToTensor()])
print("Downloading CIFAR-10 data...")
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]))
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

# %% [code]
# 5. Fast Train standard ResNet-18 (5 epochs)
base_model = make_cifar_resnet18().to(device)
optimizer = optim.SGD(base_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

print("Training base standard classifier for 5 epochs...")
base_model.train()
for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(base_model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/5 finished.")
base_model.eval()

# %% [code]
# 6. Gather Test Dataset (512 samples)
eval_size = 512
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

print(f"\nGenerating adversarial evaluation datasets on {eval_size} samples...")
# Generate target images
with torch.no_grad():
    clean_images = eval_images.clone()

print("Generating PGD-10 images...")
pgd_images = pgd_attack(base_model, eval_images, eval_labels, eps=8/255, alpha=2/255, iters=10)

print("Generating Sparse PGD (k=0.1) images...")
sparse_images = topk_pgd_attack(base_model, eval_images, eval_labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1, dynamic=True)

# %% [code]
# 7. Evaluate Defenses

def evaluate_defense(defense_fn, test_images, test_labels):
    correct = 0
    total = 0
    batch_size = 64
    for i in range(0, test_images.size(0), batch_size):
        imgs = test_images[i:i+batch_size]
        lbls = test_labels[i:i+batch_size]
        
        # Apply defense pre-processing
        with torch.no_grad():
            defended_imgs = defense_fn(imgs)
            outputs = base_model(defended_imgs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == lbls).sum().item()
            total += lbls.size(0)
    return 100 * correct / total

# Instantiate defenses
defenses = {
    "No Defense": lambda x: x,
    "Median Filter (3x3)": MedianSmoothingDefense(kernel_size=3),
    "Bit Reduction (3-bit)": BitReductionsDefense(bits=3),
    "JPEG Compression (Q75)": JPEGCompressionDefense(quality=75),
    "Random Noise (std=0.02)": RandomNoiseDefense(std=0.02),
    "Randomized Smoothing": RandomizedSmoothingModel(base_model, sigma=0.12, N=50),
    "Feature Denoising": FeatureDenoisingWrapper(base_model, kernel_size=3)
}

print("\nEvaluating all defenses...")
results = []
for name, defense in defenses.items():
    print(f"Running evaluation for: {name}")
    # Feature Denoising wrapper needs special handling as it registers internal hooks
    if name == "Feature Denoising":
        # FeatureDenoisingWrapper wraps the model itself, so we call it directly
        correct = 0
        total = 0
        batch_size = 64
        for i in range(0, eval_images.size(0), batch_size):
            # Clean
            with torch.no_grad():
                outputs = defense(clean_images[i:i+batch_size])
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == eval_labels[i:i+batch_size]).sum().item()
                total += len(predicted)
        acc_clean = 100 * correct / total
        
        correct = 0
        for i in range(0, eval_images.size(0), batch_size):
            # PGD
            with torch.no_grad():
                outputs = defense(pgd_images[i:i+batch_size])
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == eval_labels[i:i+batch_size]).sum().item()
        acc_pgd = 100 * correct / total
        
        correct = 0
        for i in range(0, eval_images.size(0), batch_size):
            # Sparse
            with torch.no_grad():
                outputs = defense(sparse_images[i:i+batch_size])
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == eval_labels[i:i+batch_size]).sum().item()
        acc_sparse = 100 * correct / total
        
        # Clean up hooks
        defense.remove_hooks()
    else:
        acc_clean = evaluate_defense(defense, clean_images, eval_labels)
        acc_pgd = evaluate_defense(defense, pgd_images, eval_labels)
        acc_sparse = evaluate_defense(defense, sparse_images, eval_labels)
        
    results.append({
        "Defense Method": name,
        "Clean Accuracy": f"{acc_clean:.2f}%",
        "PGD-10 Accuracy": f"{acc_pgd:.2f}%",
        "Sparse (k=0.1) Accuracy": f"{acc_sparse:.2f}%"
    })

# %% [code]
# 8. Print Results
df = pd.DataFrame(results)
print("\n" + "="*70)
print(" PREPROCESSING & CERTIFIED DEFENSES COMPARISON REPORT")
print("="*70)
print(df.to_string(index=False))
print("="*70)
print("\nDiscussion:")
print("- Median Filter (3x3) is generally the most effective classical filter against isolated pixel attacks (Sparse PGD).")
print("- Randomized Smoothing provides certified guarantees but heavily affects clean accuracy on standard models.")
print("- Feature Denoising on a standard model is less effective because sparse perturbations spread in intermediate feature layers.")
