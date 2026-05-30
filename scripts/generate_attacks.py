import torch
import os
import sys
import argparse
from tqdm import tqdm

# Set cache and temp directories to be inside the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

sys.path.append(workspace_dir)

from src.datasets.loader import get_cifar10, get_cifar100, get_tiny_imagenet, get_imagenet
from src.models.loader import get_model
from src.attacks.fgsm import fgsm_attack
from src.attacks.bim import bim_attack
from src.attacks.pgd import pgd_attack
from src.attacks.topk_pgd import topk_pgd_attack

def generate_attacks(dataset='cifar10', model_name='resnet18', num_batches=2, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"==================================================")
    print(f"Generating Adversarial Attacks")
    print(f"Dataset: {dataset} | Model: {model_name} | Device: {device}")
    print(f"==================================================")
    
    # 1. Load Dataset Loader
    if dataset == 'cifar10':
        loader = get_cifar10(batch_size=batch_size)
    elif dataset == 'cifar100':
        loader = get_cifar100(batch_size=batch_size)
    elif dataset == 'tiny_imagenet':
        loader = get_tiny_imagenet(batch_size=batch_size)
    elif dataset == 'imagenet':
        loader = get_imagenet(batch_size=batch_size, resize=224)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
        
    if loader is None:
        print(f"Error: Loader for {dataset} could not be initialized. Setup dataset first.")
        return
        
    # 2. Load Model
    is_robust = (model_name.lower() in ['trades', 'gg_sat']) or ('robust' in model_name.lower())
    mname = 'resnet18'
    if 'trades' in model_name.lower():
        mname = 'trades'
    elif 'gg_sat' in model_name.lower():
        mname = 'gg_sat'
        is_robust = True
    elif 'resnet50' in model_name.lower():
        mname = 'resnet50'
        
    try:
        model = get_model(mname, dataset=dataset, robust=is_robust).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model {model_name} for {dataset}: {e}. Falling back to standard ResNet18 structure.")
        import torchvision.models as models
        model = models.resnet18(num_classes=10 if 'cifar' in dataset else 1000)
        if 'cifar' in dataset:
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
        model = model.to(device)
        model.eval()

    # 3. Setup output folder
    out_dir = os.path.join(workspace_dir, 'data', 'adv_images', dataset, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # 4. Containers for batches
    attacks = {
        'Clean': [],
        'FGSM': [],
        'BIM': [],
        'PGD': [],
        'Sparse': []
    }
    
    clean_images_list = []
    labels_list = []
    
    # Run attack loops
    batch_count = 0
    for images, labels in tqdm(loader, total=num_batches, desc="Generating Attacks"):
        if batch_count >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)
        
        # Save clean images and labels
        clean_images_list.append(images.cpu())
        labels_list.append(labels.cpu())
        
        # A. Clean
        attacks['Clean'].append(images.cpu())
        
        # B. FGSM
        adv_fgsm = fgsm_attack(model, images, labels, eps=8/255)
        attacks['FGSM'].append(adv_fgsm.cpu())
        
        # C. BIM
        adv_bim = bim_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10)
        attacks['BIM'].append(adv_bim.cpu())
        
        # D. PGD
        adv_pgd = pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10)
        attacks['PGD'].append(adv_pgd.cpu())
        
        # E. Sparse (k_ratio=0.1)
        adv_sparse = topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1)
        attacks['Sparse'].append(adv_sparse.cpu())
        
        batch_count += 1
        
    # Concatenate and save to disk
    all_clean = torch.cat(clean_images_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    for attack_name, images_batches in attacks.items():
        all_adv = torch.cat(images_batches, dim=0)
        save_path = os.path.join(out_dir, f"{attack_name}.pt")
        
        # Save as dictionary
        torch.save({
            'adv_images': all_adv,
            'clean_images': all_clean,
            'labels': all_labels
        }, save_path)
        
    print(f"Successfully generated and saved adversarial attacks to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Attack Generation Stage")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "tiny_imagenet", "imagenet"],
                        help="Dataset to attack")
    parser.add_argument("--model", type=str, default="resnet18", help="Target model to attack")
    parser.add_argument("--batches", type=int, default=2, help="Number of batches to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    generate_attacks(dataset=args.dataset, model_name=args.model, num_batches=args.batches, batch_size=args.batch_size)
