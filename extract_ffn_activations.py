"""
Extract FFN first layer activations from LLaVA model on OCR-VQA dataset.
This script loads a LLaVA model and extracts the intermediate activations
after the first layer of each FFN block (gate_proj output) in the LLaMA decoder.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
import os
import numpy as np
import json

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


@torch.no_grad()
def pca_analysis(data, energy_levels=(0.90, 0.95, 0.99)):
    """
    Perform PCA analysis on the data.
    Returns the number of principal components needed to capture different energy levels.

    Args:
        data: Tensor of shape [total_tokens, hidden_dim] or [batch, seq_len, hidden_dim]
        energy_levels: Tuple of variance ratios to compute (e.g., 0.99 for 99%)

    Returns:
        Dictionary with PCA results
    """
    # Flatten to 2D if needed: [batch, seq_len, hidden_dim] -> [batch*seq_len, hidden_dim]
    if len(data.shape) == 3:
        data = data.view(-1, data.shape[-1])

    # Should be 2D now: [num_samples, hidden_dim]
    assert len(data.shape) == 2, f"Expected 2D data, got shape {data.shape}"

    # Move to CPU if needed
    if data.device.type == 'cuda':
        data = data.cpu()

    # Convert to float32 (eigh doesn't support float16)
    if data.dtype == torch.float16:
        data = data.float()

    # Center the data
    data = data - data.mean(0)

    # Compute covariance matrix
    covariance = cov(data, rowvar=False)

    # Compute eigenvalues (sorted small to large)
    eigenvalues, _ = torch.linalg.eigh(covariance)

    # Convert to numpy for easier computation
    eigenvalues = eigenvalues.numpy()

    # Compute results for each energy level
    results = {}
    total = np.sum(eigenvalues)
    total_components = len(eigenvalues)

    for energy in energy_levels:
        accum = 0
        k = 1
        while accum < energy and k <= total_components:
            accum += eigenvalues[-k] / total
            k += 1
        num_components = k - 1
        results[f'{int(energy*100)}%'] = {
            'num_components': int(num_components),
            'actual_variance': float(accum),
            'ratio': float(num_components / total_components)
        }

    return {
        'total_components': int(total_components),
        'energy_levels': results
    }


def register_ffn_hooks(model, activations_dict):
    """
    Register forward hooks to capture FFN first layer outputs (gate_proj).
    Accumulates activations from multiple forward passes.

    Args:
        model: The LLaVA model
        activations_dict: Dictionary to store list of activations

    Returns:
        List of hook handles
    """
    hooks = []

    # Access the language model layers
    # For LLaVA: model.model.layers contains the LLaMA decoder layers
    for layer_idx, layer in enumerate(model.model.layers):
        # Each layer has an MLP module with gate_proj, up_proj, down_proj
        mlp = layer.mlp

        def make_hook(layer_id):
            def hook_fn(module, input, output):
                # Accumulate activations from each forward pass
                # output shape: [batch_size, seq_len, hidden_dim]
                key = f'layer_{layer_id}_gate_proj'
                if key not in activations_dict:
                    activations_dict[key] = []
                activations_dict[key].append(output.detach().cpu())
            return hook_fn

        # Register hook on gate_proj output
        handle = mlp.gate_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)

    return hooks


def extract_activations(model_path, dataset_path, output_dir, num_samples=1, batch_size=4, device='cuda'):
    """
    Extract FFN activations from dataset examples.

    Args:
        model_path: Path to LLaVA model
        dataset_path: Path to OCR-VQA dataset
        output_dir: Directory to save results
        num_samples: Number of examples to process
        batch_size: Batch size for processing
        device: Device to run on
    """
    print(f"Loading model from {model_path}...")

    # Disable torch init for faster loading
    disable_torch_init()

    # Load model using LLaVA's builder
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device
    )

    print(f"Model loaded: {model_name}")
    print(f"Context length: {context_len}")

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)["test"]

    # Determine conversation mode
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    print(f"Using conversation mode: {conv_mode}")

    # Prepare Args class for image processing
    class Args:
        image_aspect_ratio = 'pad'
    args = Args()

    # Dictionary to accumulate activations across all samples
    accumulated_activations = {}

    print(f"\nProcessing {num_samples} examples with batch_size={batch_size}...")
    model.eval()

    # Register hooks once before all forward passes
    hooks = register_ffn_hooks(model, accumulated_activations)

    # Process in batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = range(batch_start, batch_end)

        # Prepare batch data
        batch_images = []
        batch_prompts = []

        for sample_idx in batch_indices:
            example = dataset[sample_idx]

            # Prepare input
            image = example['image'].convert('RGB')

            # Get question
            if 'question' in example:
                question = example['question']
            elif 'questions' in example:
                question = example['questions'][0]
            else:
                question = ""

            if sample_idx == 0:
                print(f"First example - Question: {question}")

            # Create conversation
            conv = conv_templates[conv_mode].copy()

            # Format prompt with image token
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            batch_images.append(image)
            batch_prompts.append(prompt)

        # Process images (batch)
        image_tensors = process_images(batch_images, image_processor, args)
        image_tensors = image_tensors.to(device, dtype=torch.float16)

        # Tokenize inputs with padding
        input_ids_list = []
        for prompt in batch_prompts:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_list.append(input_ids)

        # Pad to same length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        padded_input_ids = []
        attention_masks = []

        for input_ids in input_ids_list:
            padding_length = max_len - input_ids.shape[0]
            # Pad with tokenizer.pad_token_id (or 0 if not available)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padded_ids = torch.cat([input_ids, torch.full((padding_length,), pad_id, dtype=input_ids.dtype)])
            padded_input_ids.append(padded_ids)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.cat([torch.ones(input_ids.shape[0], dtype=torch.long),
                                       torch.zeros(padding_length, dtype=torch.long)])
            attention_masks.append(attention_mask)

        # Stack into batch tensors
        input_ids_batch = torch.stack(padded_input_ids).to(device)
        attention_mask_batch = torch.stack(attention_masks).to(device)

        if batch_start == 0:
            print(f"Batch input shape: {input_ids_batch.shape}")
            print(f"Batch image tensor shape: {image_tensors.shape}")
            print(f"Batch attention mask shape: {attention_mask_batch.shape}")

        # Forward pass - hooks will automatically capture activations
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                images=image_tensors,
                return_dict=True
            )

        print(f"Processed {batch_end}/{num_samples} samples")

    # Remove hooks after all forward passes
    for hook in hooks:
        hook.remove()

    # Concatenate all activations by flattening batch and sequence dimensions
    print(f"\nConcatenating activations from {num_samples} samples...")
    concatenated_activations = {}
    for key in sorted(accumulated_activations.keys()):
        # Each item in list is [batch_size, seq_len, hidden_dim]
        # Flatten all dimensions except last: [batch*seq_len, hidden_dim]
        flattened = [tensor.view(-1, tensor.shape[-1]) for tensor in accumulated_activations[key]]
        # Concatenate along token dimension to [total_tokens, hidden_dim]
        concatenated_activations[key] = torch.cat(flattened, dim=0)
        if key == 'layer_0_gate_proj':
            print(f"Concatenated shape for {key}: {concatenated_activations[key].shape}")

    # Print activation shapes
    print(f"\nExtracted activations for {len(concatenated_activations)} layers:")
    for key in sorted(concatenated_activations.keys()):
        print(f"  {key}: {concatenated_activations[key].shape}")

    # Perform PCA analysis on each layer
    print("\n" + "=" * 80)
    print("Performing PCA Analysis on Each Layer")
    print(f"Data from {num_samples} samples will be used for PCA")
    print("=" * 80)

    pca_results = {}
    for key in sorted(concatenated_activations.keys()):
        print(f"\nAnalyzing {key}...")
        activation = concatenated_activations[key]
        pca_result = pca_analysis(activation, energy_levels=(0.90, 0.95, 0.99))
        pca_results[key] = pca_result

        # Print results
        total = pca_result['total_components']
        print(f"  Total dimensions: {total}")
        for energy_level, stats in pca_result['energy_levels'].items():
            num_comp = stats['num_components']
            ratio = stats['ratio']
            print(f"    {energy_level} variance: {num_comp}/{total} components ({ratio*100:.2f}%)")

    # Save PCA results (no longer saving activations pt file)
    os.makedirs(output_dir, exist_ok=True)
    pca_results_path = os.path.join(output_dir, 'pca_analysis_results.json')

    print(f"\nSaving PCA results to {pca_results_path}...")
    with open(pca_results_path, 'w') as f:
        json.dump(pca_results, f, indent=2)

    print(f"PCA results file size: {os.path.getsize(pca_results_path) / 1024:.2f} KB")

    # Print summary (sorted by layer number)
    print("\n" + "=" * 80)
    print("PCA Analysis Summary (99% variance)")
    print("=" * 80)

    # Create summary list with layer info
    summary_list = []
    for key in sorted(pca_results.keys()):
        result = pca_results[key]
        num_comp_99 = result['energy_levels']['99%']['num_components']
        total = result['total_components']
        ratio = result['energy_levels']['99%']['ratio']
        layer_num = int(key.split('_')[1])
        summary_list.append({
            'layer': layer_num,
            'num_components': num_comp_99,
            'total_components': total,
            'ratio': ratio,
            'percentage': ratio * 100
        })
        print(f"Layer {layer_num:2d}: {num_comp_99:5d}/{total} components ({ratio*100:5.2f}%)")

    # Sort by percentage (ascending order)
    summary_sorted = sorted(summary_list, key=lambda x: x['percentage'])

    # Save sorted summary to file
    summary_path = os.path.join(output_dir, 'pca_summary_sorted.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PCA Analysis Summary (99% variance) - Sorted by Percentage\n")
        f.write("=" * 80 + "\n")
        for item in summary_sorted:
            f.write(f"Layer {item['layer']:2d}: {item['num_components']:5d}/{item['total_components']} components ({item['percentage']:5.2f}%)\n")

    print(f"\nSorted summary saved to {summary_path}")

    return pca_results


def main():
    # Configuration
    model_path = "/gpfs/volcano/models/llava-v1.5-7b"
    dataset_path = "/gpfs/volcano/models/howard-hou-OCR-VQA"
    output_dir = "./ffn_activations"
    num_samples = 64  # Number of samples to process
    batch_size = 64  # Batch size for processing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("FFN Activation Extraction for LLaVA Model")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Num samples: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 80)

    pca_results = extract_activations(model_path, dataset_path, output_dir, num_samples, batch_size, device)

    print("\n" + "=" * 80)
    print("Extraction Summary:")
    print(f"PCA analysis completed for all {len(pca_results)} layers")
    print(f"Expected: 32 layers (one for each Transformer layer)")
    print("=" * 80)


if __name__ == "__main__":
    main()
