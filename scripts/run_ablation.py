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
from src.utils.metrics import get_metrics

def run_ablation(num_batches=1, batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = get_cifar10(batch_size=batch_size)
    model = get_model('resnet18', dataset='cifar10', robust=False).to(device)
    model.eval()
    
    k = 0.1
    ema_scores = [0.0, 0.5, 0.9]
    dynamics = [True, False]
    
    os.makedirs('results/ablation', exist_ok=True)
    cols = ['EMA', 'Dynamic', 'Accuracy', 'ASR', 'L0', 'Sparsity']
    all_rows = []
    
    curr_num_batches = num_batches if num_batches else len(loader)
    
    for ema in ema_scores:
        for dyn in dynamics:
            print(f"Testing EMA={ema}, Dynamic={dyn}")
            total_acc, total_asr, total_l0, total_sparsity = 0, 0, 0, 0
            count = 0
            
            for batch_idx, (images, labels) in enumerate(tqdm(loader, total=curr_num_batches)):
                if batch_idx >= curr_num_batches: break
                images, labels = images.to(device), labels.to(device)
                
                adv_img = topk_pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10, k_ratio=k, dynamic=dyn, score_ema=ema)
                metrics = get_metrics(model, images, labels, adv_img, device)
                
                total_acc += metrics['acc']
                total_asr += metrics['asr']
                total_l0 += metrics['l0']
                total_sparsity += metrics['sparsity']
                count += 1
                
            all_rows.append([ema, dyn, total_acc/count, total_asr/count, total_l0/count, total_sparsity/count])
            
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv('results/ablation/ablation_results.csv', index=False)
    print("Ablation study complete. Results saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    b = 1 if args.quick else 10
    sz = 10 if args.quick else 128
    run_ablation(num_batches=b, batch_size=sz)
