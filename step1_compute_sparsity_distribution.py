"""
Step 1: Compute adaptive sparsity distribution for LLaVA model.

This script demonstrates MULTIFLOW's first step: automatically discovering
how much each layer should be pruned based on modality-specific redundancy.

Usage:
    python step1_compute_sparsity_distribution.py --target_sparsity 0.5
"""

import torch
import torch.nn as nn
import argparse
from collections import defaultdict
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


def find_linear_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find all Linear layers in a module.
    Returns dict: {layer_name: layer_module}
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_linear_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def detect_modality_llava(param_name):
    """
    Detect modality for LLaVA-v1.5 architecture.

    LLaVA structure:
    - model.vision_tower.*           -> VISION modality (CLIP ViT)
    - model.mm_projector.*           -> FUSION modality (MLP connector)
    - model.model.layers.*           -> FUSION modality (LLM decoder, processes fused vision+text)
    - model.model.embed_tokens.*    -> TEXT modality (but usually excluded from pruning)
    - model.lm_head.*               -> TEXT modality (output projection)

    Args:
        param_name: Full parameter name (e.g., "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")

    Returns:
        modality: 'vision', 'text', or 'fusion'
    """
    if 'vision_tower' in param_name:
        return 'vision'
    elif 'mm_projector' in param_name:
        return 'fusion'
    elif 'model.layers' in param_name:
        # LLM decoder processes both text and fused visual tokens
        return 'fusion'
    elif 'embed_tokens' in param_name or 'lm_head' in param_name:
        return 'text'
    else:
        # Default: fusion for unknown layers
        return 'fusion'


def compute_multimodal_distribution(model, target_sparsity=0.5, verbose=True):
    """
    Step 1 of MULTIFLOW: Compute adaptive sparsity distribution.

    This function:
    1. Groups layers by modality (vision/text/fusion)
    2. Computes magnitude-based scores for each layer
    3. Applies target sparsity to each modality separately
    4. Records the actual sparsity achieved for each layer

    Args:
        model: LLaVA model
        target_sparsity: Global target sparsity (0-1), e.g., 0.5 = 50%
        verbose: Whether to print detailed information

    Returns:
        distribution: Dict mapping layer_id -> target_sparsity for that layer
        modality_stats: Dict with statistics per modality
    """
    print("="*80)
    print("MULTIFLOW Step 1: Computing Adaptive Sparsity Distribution")
    print("="*80)
    print(f"Target global sparsity: {target_sparsity*100:.1f}%\n")

    # Find all Linear layers in the model
    linear_layers = find_linear_layers(model)
    print(f"Found {len(linear_layers)} Linear layers in the model\n")

    # Group layers by modality and compute scores
    modality_layers = defaultdict(dict)  # {modality: {layer_name: weight}}
    modality_scores = defaultdict(list)  # {modality: [flattened_scores]}

    print("Grouping layers by modality and computing magnitude scores...")
    for name, layer in linear_layers.items():
        # Detect modality
        modality = detect_modality_llava(name)

        # Get weight matrix
        W = layer.weight.data

        # Compute magnitude-based score (simple baseline, MULTIFLOW uses activations in Step 2)
        # Here we use |W| as a proxy to demonstrate the distribution mechanism
        score = torch.abs(W)

        # Store for later processing
        modality_layers[modality][name] = {
            'weight': W,
            'score': score,
            'numel': W.numel()
        }

        # Flatten and add to modality scores
        modality_scores[modality].append(score.flatten())

    # Print modality statistics
    print("\nModality-wise parameter distribution:")
    print("-" * 80)
    total_params = sum(sum(layer['numel'] for layer in layers.values())
                      for layers in modality_layers.values())

    for modality in sorted(modality_layers.keys()):
        num_layers = len(modality_layers[modality])
        num_params = sum(layer['numel'] for layer in modality_layers[modality].values())
        pct = num_params / total_params * 100
        print(f"{modality.upper():8s}: {num_layers:3d} layers, {num_params:12,d} params ({pct:5.2f}%)")
    print(f"{'TOTAL':8s}: {len(linear_layers):3d} layers, {total_params:12,d} params (100.00%)")
    print("-" * 80)

    # Step 1: Apply target sparsity to each modality SEPARATELY
    print(f"\nApplying {target_sparsity*100:.1f}% sparsity to each modality...")
    print("=" * 80)

    layer_distribution = {}  # Will store target sparsity for each layer
    modality_stats = {}  # Will store statistics per modality

    for modality in sorted(modality_layers.keys()):
        print(f"\n{'='*80}")
        print(f"Processing {modality.upper()} modality")
        print(f"{'='*80}")

        # Concatenate all scores for this modality
        all_scores = torch.cat(modality_scores[modality])
        total_elements = all_scores.numel()

        print(f"Total parameters in {modality}: {total_elements:,}")

        # Compute threshold for target sparsity
        k = int(total_elements * target_sparsity)
        threshold, _ = torch.kthvalue(all_scores.cpu(), k=k)

        print(f"Target sparsity: {target_sparsity*100:.1f}%")
        print(f"Pruning threshold: {threshold:.6f}")
        print(f"Will prune {k:,} / {total_elements:,} parameters")

        # Apply threshold to each layer in this modality and record actual sparsity
        print(f"\nPer-layer sparsity distribution in {modality.upper()}:")
        print("-" * 80)
        print(f"{'Layer Name':<80s} {'Sparsity':>10s}")
        print("-" * 80)

        modality_total_params = 0
        modality_total_pruned = 0
        layer_sparsities = []

        for layer_name, layer_info in modality_layers[modality].items():
            score = layer_info['score']

            # Compute mask based on threshold
            mask = (score > threshold)  # True = keep, False = prune

            # Calculate actual sparsity for this layer
            pruned_params = (~mask).sum().item()
            total_params_layer = score.numel()
            actual_sparsity = pruned_params / total_params_layer

            # Store in distribution dict
            layer_distribution[layer_name] = actual_sparsity
            layer_sparsities.append(actual_sparsity)

            # Accumulate statistics
            modality_total_params += total_params_layer
            modality_total_pruned += pruned_params

            # Print (show first 5 and last 5 layers to avoid clutter)
            if len(layer_sparsities) <= 5 or len(layer_sparsities) > len(modality_layers[modality]) - 5:
                # Shorten layer name for display
                display_name = layer_name if len(layer_name) <= 77 else layer_name[:74] + "..."
                print(f"{display_name:<80s} {actual_sparsity*100:>9.2f}%")
            elif len(layer_sparsities) == 6:
                print(f"{'...':<80s} {'...':>10s}")

        # Modality-level statistics
        modality_actual_sparsity = modality_total_pruned / modality_total_params
        modality_stats[modality] = {
            'num_layers': len(modality_layers[modality]),
            'total_params': modality_total_params,
            'pruned_params': modality_total_pruned,
            'actual_sparsity': modality_actual_sparsity,
            'min_layer_sparsity': min(layer_sparsities),
            'max_layer_sparsity': max(layer_sparsities),
            'mean_layer_sparsity': sum(layer_sparsities) / len(layer_sparsities),
            'std_layer_sparsity': torch.tensor(layer_sparsities).std().item(),
        }

        print("-" * 80)
        print(f"Modality summary:")
        print(f"  Layers: {len(modality_layers[modality])}")
        print(f"  Total params: {modality_total_params:,}")
        print(f"  Pruned params: {modality_total_pruned:,}")
        print(f"  Actual sparsity: {modality_actual_sparsity*100:.2f}%")
        print(f"  Per-layer sparsity range: [{min(layer_sparsities)*100:.2f}%, {max(layer_sparsities)*100:.2f}%]")
        print(f"  Per-layer sparsity mean ± std: {modality_stats[modality]['mean_layer_sparsity']*100:.2f}% ± {modality_stats[modality]['std_layer_sparsity']*100:.2f}%")

    # Global statistics
    print("\n" + "="*80)
    print("GLOBAL SUMMARY")
    print("="*80)

    global_total_params = sum(stats['total_params'] for stats in modality_stats.values())
    global_pruned_params = sum(stats['pruned_params'] for stats in modality_stats.values())
    global_actual_sparsity = global_pruned_params / global_total_params

    print(f"\nTarget global sparsity: {target_sparsity*100:.2f}%")
    print(f"Actual global sparsity: {global_actual_sparsity*100:.2f}%")
    print(f"\nModality-wise breakdown:")
    print("-" * 80)
    print(f"{'Modality':<10s} {'Params':>12s} {'Pruned':>12s} {'Sparsity':>10s} {'% of Total':>12s}")
    print("-" * 80)

    for modality in sorted(modality_stats.keys()):
        stats = modality_stats[modality]
        pct_of_total = stats['total_params'] / global_total_params * 100
        print(f"{modality.upper():<10s} "
              f"{stats['total_params']:>12,d} "
              f"{stats['pruned_params']:>12,d} "
              f"{stats['actual_sparsity']*100:>9.2f}% "
              f"{pct_of_total:>11.2f}%")

    print("-" * 80)
    print(f"{'TOTAL':<10s} "
          f"{global_total_params:>12,d} "
          f"{global_pruned_params:>12,d} "
          f"{global_actual_sparsity*100:>9.2f}% "
          f"{100.0:>11.2f}%")
    print("="*80)

    # Key insight
    print("\n" + "🔍 KEY INSIGHT ".center(80, "="))
    print("\nMULTIFLOW's adaptive sparsity distribution:")
    for modality in sorted(modality_stats.keys()):
        stats = modality_stats[modality]
        difference = (stats['actual_sparsity'] - target_sparsity) * 100
        direction = "MORE redundant" if difference > 0 else "LESS redundant"
        print(f"  • {modality.upper():8s}: {stats['actual_sparsity']*100:5.2f}% "
              f"({difference:+.2f}% vs target) → {direction}")

    print("\nThis shows which modalities have more/less redundancy!")
    print("In Step 2, we'll use activation statistics for more accurate scoring.")
    print("="*80)

    return layer_distribution, modality_stats


def main():
    parser = argparse.ArgumentParser(description='MULTIFLOW Step 1: Compute Sparsity Distribution')
    parser.add_argument('--model_path', type=str,
                        default='/gpfs/volcano/models/llava-v1.5-7b',
                        help='Path to LLaVA model')
    parser.add_argument('--target_sparsity', type=float, default=0.5,
                        help='Target global sparsity ratio (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to load model on')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MULTIFLOW Step 1: Adaptive Sparsity Distribution for LLaVA".center(80))
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Target sparsity: {args.target_sparsity*100:.1f}%")
    print(f"Device: {args.device}")

    # Load model
    print("\n" + "="*80)
    print("Loading LLaVA model...")
    print("="*80)
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=args.device
    )
    print(f"✓ Model loaded: {model_name}\n")

    # Compute sparsity distribution
    distribution, modality_stats = compute_multimodal_distribution(
        model,
        target_sparsity=args.target_sparsity,
        verbose=True
    )

    # Save results
    import json
    output_file = f'sparsity_distribution_{args.target_sparsity}.json'
    results = {
        'target_sparsity': args.target_sparsity,
        'modality_stats': {
            k: {k2: (v2 if not isinstance(v2, torch.Tensor) else v2.item())
                for k2, v2 in v.items()}
            for k, v in modality_stats.items()
        },
        'layer_distribution': {k: float(v) for k, v in distribution.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "="*80)
    print("Next step: Use these sparsity targets with activation-based scoring")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
