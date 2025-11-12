"""
Extract FFN first layer activations from LLaVA model on OCR-VQA dataset.
This script loads a LLaVA model and extracts the intermediate activations
after the first layer of each FFN block (gate_proj output) in the LLaMA decoder.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
import os

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def register_ffn_hooks(model, activations_dict):
    """
    Register forward hooks to capture FFN first layer outputs (gate_proj).

    Args:
        model: The LLaVA model
        activations_dict: Dictionary to store activations

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

    # Take first example
    example = dataset[0]

    # Prepare input
    image = example['image'].convert('RGB')

    # Get question (OCR-VQA has 'questions' field which is a list)
    if 'question' in example:
        question = example['question']
    elif 'questions' in example:
        question = example['questions'][0]
    else:
        question = ""

    print(f"\nProcessing first example:")
    print(f"Question: {question}")

    # Determine conversation mode
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    print(f"Using conversation mode: {conv_mode}")

    # Create conversation
    conv = conv_templates[conv_mode].copy()

    # Format prompt with image token
    inp = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(f"Prompt: {prompt[:200]}...")

    # Process image
    class Args:
        image_aspect_ratio = 'pad'

    args = Args()
    image_tensor = process_images([image], image_processor, args)
    image_tensor = image_tensor.to(device, dtype=torch.float16)

    # Tokenize input
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Image tensor shape: {image_tensor.shape}")

    # Dictionary to store activations
    activations_dict = {}

    # Register hooks
    print("\nRegistering forward hooks on gate_proj layers...")
    hooks = register_ffn_hooks(model, activations_dict)
    print(f"Registered {len(hooks)} hooks")

    # Forward pass
    print("Running forward pass...")
    model.eval()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            return_dict=True
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print activation shapes
    print(f"\nExtracted {len(activations_dict)} layer activations:")
    for key in sorted(activations_dict.keys()):
        print(f"  {key}: {activations_dict[key].shape}")

    # Save activations
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ffn_gate_proj_activations.pt')

    print(f"\nSaving activations to {output_path}...")
    torch.save(activations_dict, output_path)

    print(f"Done! Saved {len(activations_dict)} layer activations.")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return activations_dict


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
