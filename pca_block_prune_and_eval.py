"""
PCA-based Block Pruning for LLaVA

This script implements block-level pruning based on PCA analysis:
1. Load pre-computed PCA results from extract_ffn_activations.py
2. Rank blocks by PCA efficiency (lower percentage = less important)
3. Prune (zero-out) the least important blocks
4. Evaluate on OCR-VQA dataset

Core Idea:
- Blocks that need FEWER principal components to reach 99% variance are MORE redundant
- These redundant blocks are less important and can be pruned

Usage:
    python pca_block_prune_and_eval.py --pruning_ratio 0.5 --num_eval_samples 5000
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import re
from tqdm import tqdm
import numpy as np
from PIL import Image
from datetime import datetime

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from datasets import load_dataset


# ============================================================================
# Text normalization (from baseline code)
# ============================================================================
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
articles = ['a', 'an', 'the']
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def clean_text(pred):
    pred = pred.replace('\n', ' ')
    pred = pred.replace('\t', ' ')
    pred = pred.strip()
    pred = processPunctuation(pred)
    pred = processDigitArticle(pred)
    return pred

def compute_rouge(labels, predictions):
    """Compute ROUGE score like baseline."""
    from pycocoevalcap.rouge.rouge import Rouge

    scorer = Rouge()
    labels_dict = {}
    predictions_dict = {}

    for i in range(len(labels)):
        labels_dict[i] = [clean_text(t) for t in labels[i]]
        predictions_dict[i] = [clean_text(predictions[i])]

    (score, scores) = scorer.compute_score(labels_dict, predictions_dict)
    return score, scores


def load_pca_results(pca_results_path):
    """
    Load PCA analysis results from JSON file.

    Returns:
        pca_results: Dict with PCA results for each layer
    """
    print(f"Loading PCA results from {pca_results_path}...")

    if not os.path.exists(pca_results_path):
        raise FileNotFoundError(
            f"PCA results file not found: {pca_results_path}\n"
            f"Please run extract_ffn_activations.py first to generate PCA results."
        )

    with open(pca_results_path, 'r') as f:
        pca_results = json.load(f)

    print(f"✓ Loaded PCA results for {len(pca_results)} layers")
    return pca_results


def rank_blocks_by_pca(pca_results):
    """
    Rank blocks by PCA efficiency.

    Lower percentage = fewer components needed = more redundant = less important

    Returns:
        sorted_blocks: List of (layer_idx, percentage, num_components, total_components)
                      sorted by percentage (ascending = least important first)
    """
    print("\n" + "="*80)
    print("Ranking Blocks by PCA Efficiency")
    print("="*80)

    block_rankings = []

    for key in sorted(pca_results.keys()):
        # Extract layer number from key like "layer_0_gate_proj"
        layer_idx = int(key.split('_')[1])

        # Get 99% variance statistics
        stats_99 = pca_results[key]['energy_levels']['99%']
        num_components = stats_99['num_components']
        ratio = stats_99['ratio']
        percentage = ratio * 100

        total_components = pca_results[key]['total_components']

        block_rankings.append({
            'layer_idx': layer_idx,
            'percentage': percentage,
            'num_components': num_components,
            'total_components': total_components,
            'ratio': ratio
        })

    # Sort by percentage (ascending: lower percentage = less important)
    sorted_blocks = sorted(block_rankings, key=lambda x: x['percentage'])

    print("\nBlock Importance Ranking (99% PCA variance):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Layer':<8} {'Components':<20} {'Percentage':<12} {'Importance'}")
    print("-" * 80)

    for rank, block in enumerate(sorted_blocks):
        importance = "LOW" if rank < len(sorted_blocks) // 2 else "HIGH"
        print(f"{rank+1:<6} {block['layer_idx']:<8} "
              f"{block['num_components']}/{block['total_components']:<15} "
              f"{block['percentage']:>6.2f}%      {importance}")

    print("-" * 80)
    print(f"\nInterpretation:")
    print(f"  • Lower percentage = fewer components needed = more redundant = LESS important")
    print(f"  • Higher percentage = more components needed = less redundant = MORE important")
    print("="*80)

    return sorted_blocks


def apply_block_pruning(model, blocks_to_prune, pruning_method='zero_out'):
    """
    Apply block-level pruning by zeroing out weights.

    Args:
        model: LLaVA model
        blocks_to_prune: List of block indices to prune
        pruning_method: 'zero_out' (set all weights to 0)

    Returns:
        pruning_stats: Statistics about pruning
    """
    print("\n" + "="*80)
    print(f"Applying Block Pruning (method: {pruning_method})")
    print("="*80)

    total_blocks = len(model.model.layers)
    num_pruned = len(blocks_to_prune)

    print(f"\nTotal blocks: {total_blocks}")
    print(f"Blocks to prune: {num_pruned}")
    print(f"Pruning ratio: {num_pruned/total_blocks*100:.1f}%")
    print(f"\nPruning blocks: {sorted(blocks_to_prune)}")

    total_params = 0
    pruned_params = 0

    for layer_idx in tqdm(range(total_blocks), desc="Processing blocks"):
        layer = model.model.layers[layer_idx]

        # Count parameters in this block
        block_params = sum(p.numel() for p in layer.parameters())
        total_params += block_params

        if layer_idx in blocks_to_prune:
            # Zero out all parameters in this block
            for param in layer.parameters():
                param.data.zero_()

            pruned_params += block_params

    actual_sparsity = pruned_params / total_params if total_params > 0 else 0

    print(f"\n✓ Pruning completed!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Pruned parameters: {pruned_params:,}")
    print(f"  Actual sparsity: {actual_sparsity*100:.2f}%")

    return {
        'total_blocks': total_blocks,
        'pruned_blocks': num_pruned,
        'block_pruning_ratio': num_pruned / total_blocks,
        'total_params': total_params,
        'pruned_params': pruned_params,
        'param_sparsity': actual_sparsity,
        'pruned_block_indices': sorted(blocks_to_prune)
    }


def evaluate_ocrvqa(model, tokenizer, image_processor, dataset, num_eval_samples, device):
    """Evaluate model on OCR-VQA dataset (same as baseline)."""
    print("\n" + "="*80)
    print(f"Evaluating on OCR-VQA ({num_eval_samples} samples)")
    print("="*80)

    model_name = model.config._name_or_path
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    class Args:
        image_aspect_ratio = "pad"
    args = Args()

    model.eval()
    predictions = []
    labels = []
    skipped = 0

    for idx in tqdm(range(num_eval_samples), desc="Evaluating"):
        example = dataset[idx]

        # Image processing with error handling
        try:
            image = example["image"]
            # Skip tiny images (1x1 bad images)
            if image.size[0] < 2 or image.size[1] < 2:
                skipped += 1
                continue
            image = image.convert("RGB")
            image_tensor = process_images([image], image_processor, args)
            image_tensor = image_tensor.to(device=device, dtype=torch.float16)
        except Exception as e:
            print(f"\n[WARNING] Skipping sample {idx} due to image error: {e}")
            skipped += 1
            continue

        question = example.get("question", example.get("questions", [""])[0])
        answer = example.get("answer", example.get("answers", [""])[0])

        conv = conv_templates[conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        # Greedy decoding
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=50,
                use_cache=True,
            )

        output = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip().replace("</s>", "")

        predictions.append(output)
        labels.append([answer])

        if idx < 3:
            print(f"\nExample {idx}:")
            print(f"  Q:  {question}")
            print(f"  GT: {answer}")
            print(f"  Pred: '{output}'")

    print(f"\nSkipped {skipped} samples due to errors")
    return predictions, labels


def main():
    parser = argparse.ArgumentParser(
        description='PCA-based Block Pruning for LLaVA',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model_path', type=str,
                        default='/gpfs/volcano/models/llava-v1.5-7b',
                        help='Path to LLaVA model')
    parser.add_argument('--pca_results_path', type=str,
                        default='./ffn_activations/pca_analysis_results.json',
                        help='Path to PCA results JSON file')
    parser.add_argument('--pruning_ratio', type=float, default=0.5,
                        help='Block pruning ratio (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--dataset_path', type=str,
                        default='/gpfs/volcano/models/howard-hou-OCR-VQA',
                        help='Path to OCR-VQA dataset')
    parser.add_argument('--num_eval_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--output_dir', type=str, default='./pca_prune_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"pca_prune_{args.pruning_ratio}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("PCA-based Block Pruning for LLaVA".center(80))
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"PCA results: {args.pca_results_path}")
    print(f"Pruning ratio: {args.pruning_ratio*100:.1f}%")
    print(f"Evaluation samples: {args.num_eval_samples}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Step 1: Load PCA results
    pca_results = load_pca_results(args.pca_results_path)

    # Step 2: Rank blocks by PCA efficiency
    sorted_blocks = rank_blocks_by_pca(pca_results)

    # Step 3: Determine which blocks to prune
    # Use the actual number of blocks from PCA results (not hardcoded)
    total_blocks = len(sorted_blocks)
    num_blocks_to_prune = int(total_blocks * args.pruning_ratio)

    # Prune the least important blocks (lowest percentages)
    blocks_to_prune = [block['layer_idx'] for block in sorted_blocks[:num_blocks_to_prune]]

    print(f"\n" + "="*80)
    print(f"Pruning Strategy")
    print("="*80)
    print(f"Total blocks (from PCA results): {total_blocks}")
    print(f"Target pruning ratio: {args.pruning_ratio*100:.1f}%")
    print(f"Blocks to prune: {num_blocks_to_prune}")
    print(f"Pruned blocks (least important): {sorted(blocks_to_prune)}")
    print("="*80)

    # Step 4: Load model
    print("\nLoading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=args.device
    )
    print(f"✓ Model loaded: {model_name}\n")

    # Step 5: Apply block pruning
    pruning_stats = apply_block_pruning(model, blocks_to_prune, pruning_method='zero_out')

    # Step 6: Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset_path)["test"]
    print(f"✓ Dataset loaded: {len(dataset)} total samples")

    # Step 7: Evaluate
    if args.num_eval_samples is None:
        num_eval_samples = len(dataset)
    else:
        num_eval_samples = min(args.num_eval_samples, len(dataset))

    predictions, labels = evaluate_ocrvqa(
        model, tokenizer, image_processor, dataset, num_eval_samples, args.device
    )

    # Step 8: Compute ROUGE
    rouge_score, rouge_scores = compute_rouge(labels, predictions)

    print("\n" + "="*80)
    print("Evaluation Results:")
    print("="*80)
    print(f"Method: PCA-based Block Pruning")
    print(f"Total samples evaluated: {len(predictions)}")
    print(f"Block pruning ratio: {pruning_stats['block_pruning_ratio']*100:.2f}%")
    print(f"Parameter sparsity: {pruning_stats['param_sparsity']*100:.2f}%")
    print(f"ROUGE score: {rouge_score:.4f}")
    print("="*80)

    # Step 9: Save results
    results_dict = {
        "predictions": predictions,
        "labels": labels,
        "rouge_scores": rouge_scores.tolist()
    }
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    summary = {
        "model": args.model_path,
        "dataset": args.dataset_path,
        "method": "PCA-based Block Pruning",
        "pca_results_path": args.pca_results_path,
        "pruning_ratio": args.pruning_ratio,
        "pruning_stats": pruning_stats,
        "num_eval_samples": len(predictions),
        "rouge_score": float(rouge_score),
        "timestamp": timestamp,
        "pruned_blocks": sorted(blocks_to_prune),
        "block_rankings": sorted_blocks
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - {eval_path}")
    print(f"  - {summary_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
