import sys
import os

workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

import torch
import pandas as pd
from tqdm import tqdm

sys.path.append(workspace_dir)

from src.datasets.loader import get_cifar10
from src.models.loader import get_model
from src.attacks.topk_pgd import topk_pgd_attack

def run_classwise(num_batches=1, batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = get_cifar10(batch_size=batch_size)
    model = get_model('resnet18', dataset='cifar10', robust=False).to(device)
    model.eval()
    
    num_classes = 10
    class_totals = torch.zeros(num_classes, device=device)
    class_successes = torch.zeros(num_classes, device=device)
    
    curr_num_batches = num_batches if num_batches else len(loader)
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader, total=curr_num_batches)):
        if batch_idx >= curr_num_batches: break
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, clean_preds = torch.max(outputs, 1)
            correct_idx = (clean_preds == labels)
            
        adv_img = topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=0.1)
        
        with torch.no_grad():
            adv_outputs = model(adv_img)
            _, adv_preds = torch.max(adv_outputs, 1)
            
        for i in range(len(labels)):
            if correct_idx[i]:
                c = labels[i].item()
                class_totals[c] += 1
                if adv_preds[i] != labels[i]:
                    class_successes[c] += 1
                    
    asr_per_class = class_successes / (class_totals + 1e-8)
    
    os.makedirs('results/classwise', exist_ok=True)
    cols = ['Class', 'Total Correct', 'Total Fooled', 'ASR']
    rows = []
    
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(num_classes):
        rows.append([class_names[i], int(class_totals[i].item()), int(class_successes[i].item()), asr_per_class[i].item()])
        
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv('results/classwise/classwise_asr.csv', index=False)
    print("Class-wise analysis complete.")
    print(df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    
    run_classwise(num_batches=args.batches, batch_size=args.batch_size)
