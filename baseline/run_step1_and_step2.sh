#!/bin/bash
# MULTIFLOW Step 1 & 2: Score Computation
export CUDA_VISIBLE_DEVICES=7

echo "=========================================="
echo "MULTIFLOW Step 1 & 2: Score Computation"
echo "=========================================="
echo ""

# Compute sparsity distribution and information flow scores
echo "Running MULTIFLOW score computation..."
python multiflow_step1_step2_compute_scores.py \
    --target_sparsity 0.5 \
    --nsamples 32 \
    --model_path /gpfs/volcano/models/llava-v1.5-7b \
    --dataset_path /gpfs/volcano/models/howard-hou-OCR-VQA \
    --device cuda

echo ""
echo "=========================================="
echo "MULTIFLOW Step 1 & 2 completed!"
echo "Check the generated JSON file:"
echo "  - multiflow_scores_0.5.json"
echo ""
echo "This file will be used by Step 3 for pruning."
echo "=========================================="
