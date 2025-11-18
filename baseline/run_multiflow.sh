#!/bin/bash
# MULTIFLOW Pruning and Evaluation for LLaVA
export CUDA_VISIBLE_DEVICES=7

echo "=========================================="
echo "MULTIFLOW: Prune and Evaluate LLaVA"
echo "=========================================="
echo ""

# Run MULTIFLOW with 50% sparsity
# - 32 calibration samples for pruning
# - 5000 test samples for evaluation
python multiflow_prune_and_eval.py \
    --target_sparsity 0.5 \
    --nsamples 32 \
    --num_eval_samples 5000 \
    --model_path /gpfs/volcano/models/llava-v1.5-7b \
    --dataset_path /gpfs/volcano/models/howard-hou-OCR-VQA \
    --device cuda \
    --output_dir ./multiflow_results

echo ""
echo "=========================================="
echo "MULTIFLOW completed!"
echo "Check ./multiflow_results/ for outputs"
echo "=========================================="
