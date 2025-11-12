"""
Extract FFN first layer activations from LLaVA model on OCR-VQA dataset.
This script loads a LLaVA model and extracts the intermediate activations
after the first layer of each FFN block (gate_proj output) in the LLaMA decoder.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
import os
from tqdm import tqdm
import numpy as np


def register_ffn_hooks(model, activations_dict):
    """
    Register forward hooks to capture FFN first layer outputs.
    For LLaMA, the FFN structure is: gate_proj + up_proj -> silu -> down_proj
    We want to capture the output after gate_proj (before activation).

    Args:
        model: The LLaVA model
        activations_dict: Dictionary to store activations

    Returns:
        List of hook handles
    """
    hooks = []

    # Access the language model layers
    language_model = model.language_model

    # LLaMA layers are in model.layers
    for layer_idx, layer in enumerate(language_model.model.layers):
        # Each layer has an MLP module with gate_proj, up_proj, down_proj
        mlp = layer.mlp

        def make_hook(layer_id):
            def hook_fn(module, input, output):
                # Store the activation (detach to avoid keeping computation graph)
                # output shape: [batch_size, seq_len, hidden_dim]
                activations_dict[f'layer_{layer_id}_gate_proj'] = output.detach().cpu()
            return hook_fn

        # Register hook on gate_proj output
        handle = mlp.gate_proj.register_forward_hook(make_hook(layer_idx))
        hooks.append(handle)

    return hooks


def extract_activations(model_path, dataset_path, output_dir, device='cuda'):
    """
    Extract FFN activations from one example.

    Args:
        model_path: Path to LLaVA model
        dataset_path: Path to OCR-VQA dataset
        output_dir: Directory to save activations
        device: Device to run on
    """
    print(f"Loading model from {model_path}...")

    # Load model and processor
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"Loading dataset from {dataset_path}...")
    # Load dataset
    dataset = load_dataset(dataset_path)["test"]

    # Take first example
    example = dataset[0]
    print(f"\nProcessing example:")
    print(f"Question: {example.get('question', 'N/A')}")
    print(f"Answer: {example.get('answers', 'N/A')}")

    # Prepare input
    # OCR-VQA typically has 'image', 'question', 'answers' fields
    image = example['image']
    question = example.get('question', '')

    # Format prompt for LLaVA
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"\nInput shape: {inputs['input_ids'].shape}")

    # Dictionary to store activations
    activations_dict = {}

    # Register hooks
    print("Registering forward hooks...")
    hooks = register_ffn_hooks(model, activations_dict)

    # Forward pass
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print activation shapes
    print(f"\nExtracted {len(activations_dict)} layer activations:")
    for key in sorted(activations_dict.keys()):
        print(f"{key}: {activations_dict[key].shape}")

    # Save activations
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ffn_activations_layer0.pt')

    # Convert to numpy for smaller file size
    activations_np = {k: v.numpy() for k, v in activations_dict.items()}

    print(f"\nSaving activations to {output_path}...")
    torch.save(activations_np, output_path)

    print(f"Done! Saved {len(activations_np)} layer activations.")
    print(f"Total file size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return activations_np


def main():
    # Configuration
    model_path = "/gpfs/volcano/models/llava-v1.5-7b"
    dataset_path = "/gpfs/volcano/models/howard-hou-OCR-VQA"
    output_dir = "./ffn_activations"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("FFN Activation Extraction for LLaVA Model")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    activations = extract_activations(model_path, dataset_path, output_dir, device)

    print("\n" + "=" * 80)
    print("Extraction Summary:")
    print(f"Number of layers: {len(activations)}")
    print("Expected: 32 layers (one for each Transformer layer)")
    print("=" * 80)


if __name__ == "__main__":
    main()
