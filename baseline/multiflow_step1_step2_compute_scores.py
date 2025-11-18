"""
MULTIFLOW Pruning for LLaVA: Step 1 & Step 2 Score Computation

Step 1: Compute adaptive sparsity distribution (modality-aware)
Step 2: Compute information flow scores with activation statistics

This script computes and saves:
1. Layer-wise sparsity distribution (how much to prune per layer)
2. Information flow scores (which parameters to prune)

These results will be used by Step 3 for actual pruning.

Usage:
    python multiflow_step1_step2_compute_scores.py --target_sparsity 0.5 --nsamples 128
"""

import torch
import torch.nn as nn
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# For loading calibration data
from datasets import load_dataset


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


# ============================================================================
# Step 2: Information Flow Scoring with Activation Statistics
# ============================================================================

def prepare_calibration_data_llava(model, tokenizer, image_processor, dataset, nsamples, device):
    """
    Prepare calibration data from OCR-VQA dataset for MULTIFLOW pruning.

    This is similar to Wanda's calibration, but stores data in a format
    suitable for MULTIFLOW's activation collection.

    Args:
        model: LLaVA model
        tokenizer: Tokenizer
        image_processor: Image processor
        dataset: Dataset to sample from (e.g., OCR-VQA)
        nsamples: Number of calibration samples
        device: Device to run on

    Returns:
        calibration_data: List of dicts with 'input_ids', 'images', 'attention_mask'
    """
    print(f"Preparing {nsamples} calibration samples for MULTIFLOW...")

    # Determine conversation mode
    model_name = model.config._name_or_path
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    class Args:
        image_aspect_ratio = 'pad'
    args = Args()

    calibration_data = []
    for idx in tqdm(range(nsamples), desc="Preparing calibration data"):
        example = dataset[idx]

        # Process image (force RGB conversion)
        image = example['image']
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        image = Image.fromarray(img_array.astype('uint8'), mode='RGB')

        # Get question
        if 'question' in example:
            question = example['question']
        elif 'questions' in example:
            question = example['questions'][0]
        else:
            question = ""

        # Create conversation
        conv = conv_templates[conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process image
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(device, dtype=torch.float16)

        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        # Create attention mask (1 for valid tokens, 0 for padding)
        # For LLaVA, we need to track which tokens are valid (not [PAD])
        attention_mask = torch.ones_like(input_ids)

        calibration_data.append({
            'input_ids': input_ids,
            'images': image_tensor,
            'attention_mask': attention_mask,
            'question': question  # For debugging
        })

    print(f"✓ Calibration data prepared: {len(calibration_data)} samples")
    return calibration_data


class ActivationCollector:
    """
    Hook-based activation collector for MULTIFLOW.

    This class attaches hooks to Linear layers to collect input activations,
    following MULTIFLOW's approach of storing input history for later processing.
    """

    def __init__(self, layer, layer_name, modality):
        """
        Args:
            layer: The nn.Linear layer to monitor
            layer_name: Name of the layer (for debugging)
            modality: Modality of this layer ('vision', 'text', 'fusion')
        """
        self.layer = layer
        self.layer_name = layer_name
        self.modality = modality
        self.input_history = []  # Store inputs from all batches
        self.nsamples = 0

    def hook(self, module, input, output):
        """
        Forward hook to capture input activations.

        Args:
            module: The layer module
            input: Tuple of input tensors
            output: Output tensor
        """
        # Get the input tensor
        inp = input[0].data

        # Store input activation
        # Note: We clone and detach to avoid keeping computation graph
        self.input_history.append(inp.clone().detach())

    def clear(self):
        """Clear stored activations to free memory."""
        self.input_history = []
        self.nsamples = 0


def compute_activation_norms(linear_layers, calibration_data, model, device):
    """
    Step 2.1-2.3: Collect activation statistics via forward passes.

    This function:
    1. Attaches hooks to all Linear layers
    2. Runs forward passes on calibration data
    3. Computes L2 norms of input activations for each layer

    Following MULTIFLOW's implementation in multiflow.py:166-224

    Args:
        linear_layers: Dict of {layer_name: layer_module}
        calibration_data: List of calibration samples
        model: LLaVA model
        device: Device to run on

    Returns:
        activation_norms: Dict of {layer_name: activation_norm_vector}
    """
    print("\n" + "="*80)
    print("MULTIFLOW Step 2: Computing Activation Statistics")
    print("="*80)
    print(f"Collecting activations from {len(calibration_data)} samples...\n")

    # Initialize activation collectors for each layer
    collectors = {}
    handles = []

    for layer_name, layer in linear_layers.items():
        modality = detect_modality_llava(layer_name)
        collector = ActivationCollector(layer, layer_name, modality)
        collectors[layer_name] = collector

        # Register forward hook
        handle = layer.register_forward_hook(collector.hook)
        handles.append(handle)

    # Run forward passes to collect activations
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(calibration_data, desc="Collecting activations")):
            # Forward pass
            _ = model(
                input_ids=sample['input_ids'],
                images=sample['images'],
                return_dict=True
            )

            # Periodically clear GPU cache
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    # Remove all hooks
    for handle in handles:
        handle.remove()

    print("\n✓ Activation collection completed!")
    print("\nComputing activation norms for each layer...")

    # Compute activation norms (following MULTIFLOW's _offload_actns method)
    activation_norms = {}

    for layer_name, collector in tqdm(collectors.items(), desc="Computing norms"):
        if len(collector.input_history) == 0:
            print(f"Warning: No activations collected for {layer_name}")
            # Initialize to ones as fallback
            W = linear_layers[layer_name].weight.data
            activation_norms[layer_name] = torch.ones(W.shape[1], dtype=torch.float32)
            continue

        # Concatenate all input activations
        # Each input has shape [batch, seq_len, in_features]
        # We need to reshape to [total_tokens, in_features]
        all_inputs = []
        for inp in collector.input_history:
            if len(inp.shape) == 3:
                # [batch, seq, features] -> [batch*seq, features]
                inp_reshaped = inp.reshape(-1, inp.shape[-1])
            elif len(inp.shape) == 2:
                # [batch, features] -> already correct
                inp_reshaped = inp
            else:
                print(f"Warning: Unexpected input shape {inp.shape} for {layer_name}")
                continue
            all_inputs.append(inp_reshaped)

        if len(all_inputs) == 0:
            W = linear_layers[layer_name].weight.data
            activation_norms[layer_name] = torch.ones(W.shape[1], dtype=torch.float32)
            continue

        # Concatenate along batch dimension
        X = torch.cat(all_inputs, dim=0)  # [total_tokens, in_features]

        # Compute L2 norm for each input feature (following MULTIFLOW line 146)
        # actn_norms[id(param)] += torch.norm(X, p=2, dim=0) ** 2 / nsamples
        nsamples = X.shape[0]
        actn_norm = torch.norm(X.cpu().float(), p=2, dim=0) ** 2 / nsamples

        activation_norms[layer_name] = actn_norm

        # Clear memory
        collector.clear()
        del X
        torch.cuda.empty_cache()

    print(f"\n✓ Activation norms computed for {len(activation_norms)} layers")

    return activation_norms


def compute_information_flow_scores(linear_layers, activation_norms):
    """
    Step 2.4-2.5: Compute MULTIFLOW's information flow scores.

    For each layer, compute:
    1. Output neuron importance: (|W| * actn_norm).mean(dim=1)
    2. Input neuron importance: (|W| * actn_norm).mean(dim=0)
    3. Information flow score: outer(Imp_out, Imp_in) * |W|

    Following MULTIFLOW's score() method in multiflow.py:81-93

    Args:
        linear_layers: Dict of {layer_name: layer_module}
        activation_norms: Dict of {layer_name: activation_norm_vector}

    Returns:
        information_flow_scores: Dict of {layer_name: score_matrix}
    """
    print("\n" + "="*80)
    print("Computing Information Flow Scores (MULTIFLOW Formula)")
    print("="*80)
    print("\nFormula: Score = Imp(output) ⊗ Imp(input) ⊙ |W|")
    print("  where:")
    print("    Imp(output) = (|W| ⊙ actn_norm).mean(dim=1)")
    print("    Imp(input)  = (|W| ⊙ actn_norm).mean(dim=0)")
    print("    ⊗ = outer product, ⊙ = element-wise product")
    print("="*80 + "\n")

    information_flow_scores = {}

    for layer_name, layer in tqdm(linear_layers.items(), desc="Computing scores"):
        # Get weight matrix [out_features, in_features]
        W = layer.weight.data

        # Get activation norm vector [in_features]
        actn_norm = torch.sqrt(activation_norms[layer_name]).to(W.device)

        # Step 2.4: Compute neuron importance (MULTIFLOW line 85-86)
        # Importance per output neuron [out_features]
        importance_per_output = (W.abs() * actn_norm).mean(dim=1)

        # Importance per input neuron [in_features]
        importance_per_input = (W.abs() * actn_norm).mean(dim=0)

        # Step 2.5: Information flow score (MULTIFLOW line 89-92)
        # Outer product: [out_features, in_features]
        score = torch.outer(importance_per_output, importance_per_input)

        # Final score: multiply by weight magnitude
        final_score = score * W.abs()

        information_flow_scores[layer_name] = final_score.cpu()

        # Print example for first few layers
        if len(information_flow_scores) <= 3:
            print(f"\n  Layer: {layer_name}")
            print(f"    Weight shape: {W.shape}")
            print(f"    Activation norm range: [{actn_norm.min():.4f}, {actn_norm.max():.4f}]")
            print(f"    Output importance range: [{importance_per_output.min():.4f}, {importance_per_output.max():.4f}]")
            print(f"    Input importance range: [{importance_per_input.min():.4f}, {importance_per_input.max():.4f}]")
            print(f"    Final score range: [{final_score.min():.4f}, {final_score.max():.4f}]")

    print(f"\n✓ Information flow scores computed for {len(information_flow_scores)} layers")

    return information_flow_scores




def main():
    parser = argparse.ArgumentParser(
        description='MULTIFLOW Step 1 & 2: Compute Scores for LLaVA Pruning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute scores with 50% sparsity target
  python multiflow_step1_step2_compute_scores.py --target_sparsity 0.5 --nsamples 128

  # Compute scores with 70% sparsity target
  python multiflow_step1_step2_compute_scores.py --target_sparsity 0.7 --nsamples 256
        """
    )
    parser.add_argument('--model_path', type=str,
                        default='/gpfs/volcano/models/llava-v1.5-7b',
                        help='Path to LLaVA model')
    parser.add_argument('--target_sparsity', type=float, default=0.5,
                        help='Target global sparsity ratio (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to load model on')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples (default: 128)')
    parser.add_argument('--dataset_path', type=str, default='/gpfs/volcano/models/howard-hou-OCR-VQA',
                        help='Path to OCR-VQA dataset for calibration')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MULTIFLOW Step 1 & 2: Score Computation".center(80))
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Target sparsity: {args.target_sparsity*100:.1f}%")
    print(f"Device: {args.device}")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Dataset: {args.dataset_path}")
    print("="*80)

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

    # Get all linear layers
    linear_layers = find_linear_layers(model)
    print(f"Found {len(linear_layers)} Linear layers")

    # ========================================================================
    # Step 1: Compute adaptive sparsity distribution
    # ========================================================================
    print("\n" + "🔹"*40)
    print("Step 1: Computing Adaptive Sparsity Distribution")
    print("🔹"*40)

    layer_distribution, modality_stats = compute_multimodal_distribution(
        model,
        target_sparsity=args.target_sparsity,
        verbose=True
    )

    # ========================================================================
    # Step 2: Compute information flow scores with activation statistics
    # ========================================================================
    print("\n\n" + "🔹"*40)
    print("Step 2: Computing Information Flow Scores")
    print("🔹"*40)

    # Load calibration dataset
    print("\nLoading calibration dataset...")
    try:
        dataset = load_dataset(args.dataset_path)["test"]
        print(f"✓ Loaded dataset: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Prepare calibration data
    calibration_data = prepare_calibration_data_llava(
        model, tokenizer, image_processor, dataset, args.nsamples, args.device
    )

    # Compute activation norms
    activation_norms = compute_activation_norms(
        linear_layers, calibration_data, model, args.device
    )

    # Compute information flow scores
    information_flow_scores = compute_information_flow_scores(
        linear_layers, activation_norms
    )

    # ========================================================================
    # Save unified results for Step 3
    # ========================================================================
    print("\n" + "="*80)
    print("Saving Results for Step 3")
    print("="*80)

    import json

    # Convert information_flow_scores to serializable format
    scores_serializable = {}
    for layer_name, score_tensor in information_flow_scores.items():
        # Save shape and flattened values
        scores_serializable[layer_name] = {
            'shape': list(score_tensor.shape),
            'values': score_tensor.flatten().tolist()
        }

    # Unified output file
    output_file = f'multiflow_scores_{args.target_sparsity}.json'
    results = {
        'method': 'multiflow_step1_step2',
        'target_sparsity': args.target_sparsity,
        'nsamples': args.nsamples,
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,

        # Step 1 results: HOW MUCH to prune per layer
        'layer_distribution': {k: float(v) for k, v in layer_distribution.items()},

        # Step 1 statistics
        'modality_stats': {
            k: {k2: (v2 if not isinstance(v2, torch.Tensor) else v2.item())
                for k2, v2 in v.items()}
            for k, v in modality_stats.items()
        },

        # Step 2 results: WHICH parameters to prune (information flow scores)
        'information_flow_scores': scores_serializable
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nThis file contains:")
    print(f"  1. layer_distribution: Target sparsity for each layer (Step 1)")
    print(f"  2. information_flow_scores: Importance scores for pruning (Step 2)")
    print(f"  3. modality_stats: Statistics per modality")

    # Final summary
    print("\n\n" + "="*80)
    print("✅ MULTIFLOW STEP 1 & 2 COMPLETED")
    print("="*80)
    print(f"\n📁 Output file: {output_file}")
    print(f"\n📌 Next steps:")
    print(f"  → Implement Step 3 to load this file and apply pruning")
    print(f"  → Step 3 will use:")
    print(f"      • layer_distribution (how much to prune per layer)")
    print(f"      • information_flow_scores (which parameters to prune)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
