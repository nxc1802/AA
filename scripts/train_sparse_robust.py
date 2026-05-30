import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys
import time
import random
import argparse
import numpy as np

# Set cache and temp directories to be inside the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

sys.path.append(workspace_dir)

from src.attacks.topk_pgd import topk_pgd_attack
from src.utils.metrics import get_metrics

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

    # 3. Optimizer & Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

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
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Select random k_ratio for this batch
            batch_k_ratio = random.uniform(args.k_min, args.k_max)
            
            # 1) Generate Sparse Adversarial Perturbations (Inner maximization)
            # Use 5-step Top-K PGD attack to speed up training
            model.eval() # Disable dropout/batchnorm during attack generation
            adv_images = topk_pgd_attack(
                model, images, labels, 
                eps=8/255, alpha=2/255, 
                iters=5, k_ratio=batch_k_ratio, 
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
            else:
                # Mixed Training (Clean + Adv)
                outputs_clean = model(images)
                outputs_adv = model(adv_images)
                loss_clean = criterion(outputs_clean, labels)
                loss_adv = criterion(outputs_adv, labels)
                loss = (1 - args.beta) * loss_clean + args.beta * loss_adv
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * labels.size(0)
            
        scheduler.step()
        epoch_duration = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader.dataset)
        
        # 5. Fast Evaluation loop after epoch
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

    print("==================================================")
    print("Training finished successfully!")
    print(f"Best Val Robust Acc (k=0.3): {best_robust_acc*100:.2f}%")
    print(f"Best Val Clean Acc: {best_clean_acc*100:.2f}%")
    print(f"Weights saved at: {best_checkpoint_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
