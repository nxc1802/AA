import os
import re
import json

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def extract_imports_and_code(file_contents):
    imports = []
    code_lines = []
    
    for line in file_contents.split('\n'):
        # Ignore local src imports
        if re.match(r'^(from|import)\s+src\.', line):
            continue
        # Collect standard and third-party imports
        if re.match(r'^(import|from)\s+[a-zA-Z0-9_]+', line):
            imports.append(line)
        else:
            code_lines.append(line)
            
    return imports, '\n'.join(code_lines)

def generate_monolithic_script(target_script, core_files, output_py_path, output_ipynb_path):
    all_imports = set()
    all_code = []
    
    # 1. Process Core Files (Dependencies)
    for file in core_files:
        if not os.path.exists(file): continue
        content = read_file(file)
        imp, code = extract_imports_and_code(content)
        all_imports.update(imp)
        all_code.append(f"\n# --- Extracted from {file} ---")
        all_code.append(code)
        
    # 2. Process Target Script (Main Execution)
    if os.path.exists(target_script):
        content = read_file(target_script)
        imp, code = extract_imports_and_code(content)
        all_imports.update(imp)
        
        # Replace argparse with hardcoded Kaggle config block
        code = re.sub(r'def parse_args\(\):[\s\S]*?if __name__ == "__main__":', 
                      'if __name__ == "__main__":\n    # Kaggle default args\n    class Args:\n        dataset="cifar10"\n        model="resnet18"\n        batches=1\n        batch_size=16\n        epochs=10\n        mock=False\n    args = Args()', code)
        
        all_code.append(f"\n# --- Extracted from {target_script} ---")
        all_code.append(code)
    
    # Ensure some essential imports are present
    all_imports.add("import os")
    all_imports.add("import torch")
    all_imports.add("import torchvision")
    all_imports.add("import matplotlib.pyplot as plt")
    all_imports.add("import pandas as pd")
    
    # Filter bad imports
    cleaned_imports = []
    for imp in all_imports:
        if "robustbench.utils" in imp or "lpips" in imp or "torch" in imp or "torchvision" in imp or "numpy" in imp or "os" in imp or "sys" in imp or "argparse" in imp or "matplotlib" in imp or "tqdm" in imp or "typing" in imp or "csv" in imp or "pandas" in imp or "PIL" in imp or "cv2" in imp or "math" in imp or "time" in imp or "json" in imp or "random" in imp:
            cleaned_imports.append(imp)
    
    # Special pip installs for kaggle
    kaggle_header = "# %% [markdown]\n# # Kaggle Notebook\n# Run the cell below to install requirements\n# %% [code]\n!pip install robustbench lpips advertorch opencv-python\n"
    
    final_py_content = kaggle_header + "\n# %% [code]\n# ================= IMPORTS =================\n"
    final_py_content += '\n'.join(sorted(list(cleaned_imports)))
    final_py_content += "\n\n# ================= CORE MODULES =================\n"
    final_py_content += '\n'.join(all_code)
    
    # Write Python file
    os.makedirs(os.path.dirname(output_py_path), exist_ok=True)
    with open(output_py_path, 'w') as f:
        f.write(final_py_content)
        
    # Write Jupyter Notebook format
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Kaggle Notebook\n", "Generated monolithic script for AA Project."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!pip install robustbench lpips advertorch opencv-python"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in ('\n'.join(sorted(list(cleaned_imports)))).split('\n')]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in ('\n'.join(all_code)).split('\n')]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(output_ipynb_path, 'w') as f:
        json.dump(notebook, f, indent=1)
        
    print(f"Generated {output_py_path} and {output_ipynb_path}")

if __name__ == "__main__":
    core_files = [
        "src/utils/config.py",
        "src/utils/metrics.py",
        "src/utils/visualization.py",
        "src/models/gg_sat.py",
        "src/attacks/topk_pgd.py",
        "src/attacks/sparse_pgd.py",
        "src/attacks/sparsefool.py",
        "src/attacks/greedyfool.py",
    ]
    
    # 1. Final Benchmark Notebook
    generate_monolithic_script(
        target_script="scripts/run_final_bench.py",
        core_files=core_files,
        output_py_path="kaggle_notebooks/Kaggle_1_Final_Benchmark.py",
        output_ipynb_path="kaggle_notebooks/Kaggle_1_Final_Benchmark.ipynb"
    )
    
    # 2. Defense Bench Notebook
    generate_monolithic_script(
        target_script="scripts/run_defense_bench.py",
        core_files=core_files,
        output_py_path="kaggle_notebooks/Kaggle_2_Defense_Eval.py",
        output_ipynb_path="kaggle_notebooks/Kaggle_2_Defense_Eval.ipynb"
    )
    
    # 3. Train GG SAT
    generate_monolithic_script(
        target_script="scripts/train_sparse_robust.py",
        core_files=core_files,
        output_py_path="kaggle_notebooks/Kaggle_3_Train_GGSAT.py",
        output_ipynb_path="kaggle_notebooks/Kaggle_3_Train_GGSAT.ipynb"
    )
    
    # 4. Ablation Analysis
    generate_monolithic_script(
        target_script="scripts/run_ablation.py",
        core_files=core_files,
        output_py_path="kaggle_notebooks/Kaggle_4_Ablation_Analysis.py",
        output_ipynb_path="kaggle_notebooks/Kaggle_4_Ablation_Analysis.ipynb"
    )
    
    print("All monolithic notebooks generated successfully in kaggle_notebooks/")
