#!/bin/bash

# Final Pipeline Execution Script
# This script runs the FULL evaluation (75 samples) across all attack methods.
# Hessian-Patch is run first to ensure early success before the long PGD-200 run.

mkdir -p output_final_report

echo "[*] Starting Final Full Pipeline (75 Samples)..."
echo "[*] This may take ~1.5 - 2 hours due to PGD-200."
echo "[*] Results will be saved to 'output_final_report/'"

source venv/bin/activate

# Unified evaluation script handles Hessian-Patch, FGSM, and PGD-200 sequentially.
python3 scripts/final_unified_eval.py 2>&1 | tee output_final_report/pipeline_log.txt

echo "[*] Pipeline Complete. Check 'output_final_report/final_comparison.json' for results."
