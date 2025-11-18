#!/bin/bash
# PCA-based Block Pruning for LLaVA
export CUDA_VISIBLE_DEVICES=7

echo "=========================================="
echo "PCA-based Block Pruning"
echo "=========================================="
echo ""

# Step 1: Check if PCA results exist, if not, generate them
if [ ! -f "./ffn_activations/pca_analysis_results.json" ]; then
    echo "PCA results not found. Generating PCA analysis first..."
    python extract_ffn_activations.py
    echo ""
fi

# Step 2: Run PCA-based block pruning and evaluation
echo "Running PCA-based block pruning (50% pruning ratio)..."
python pca_block_prune_and_eval.py \
    --pruning_ratio 0.5 \
    --num_eval_samples 5000 \
    --model_path /gpfs/volcano/models/llava-v1.5-7b \
    --dataset_path /gpfs/volcano/models/howard-hou-OCR-VQA \
    --pca_results_path ./ffn_activations/pca_analysis_results.json \
    --device cuda \
    --output_dir ./pca_prune_results

echo ""
echo "=========================================="
echo "PCA Block Pruning completed!"
echo "Check ./pca_prune_results/ for outputs"
echo "=========================================="
