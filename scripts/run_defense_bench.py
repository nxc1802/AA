import sys
import os
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set cache and temp directories to be inside the workspace
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['TORCH_HOME'] = os.path.join(workspace_dir, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(workspace_dir, '.cache')

sys.path.append(workspace_dir)

from src.models.loader import get_model
from src.defenses.preprocessing import (
    MedianSmoothingDefense,
    BitReductionsDefense,
    JPEGCompressionDefense,
    RandomNoiseDefense,
    RandomizedSmoothingModel,
    FeatureDenoisingWrapper
)
from scripts.generate_attacks import generate_attacks

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
                "Sparse Transfer ASR": acc_sparse_t
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
