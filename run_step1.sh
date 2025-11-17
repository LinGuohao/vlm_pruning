#!/bin/bash
# Quick test script for MULTIFLOW Step 1
export CUDA_VISIBLE_DEVICES=7

echo "=========================================="
echo "MULTIFLOW Step 1: Sparsity Distribution"
echo "=========================================="
echo ""

# Test with 50% sparsity (default)
echo "Testing with 50% sparsity..."
python step1_compute_sparsity_distribution.py \
    --target_sparsity 0.5 \
    --model_path /gpfs/volcano/models/llava-v1.5-7b \
    --device cuda

echo ""
echo "=========================================="
echo "Step 1 completed!"
echo "Check the output above to see:"
echo "  - Vision modality sparsity"
echo "  - Fusion modality sparsity"
echo "  - Text modality sparsity"
echo "=========================================="
