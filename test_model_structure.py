"""
Test script to verify LLaVA model structure and FFN layers.
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def inspect_model_structure(model_path, device='cuda'):
    """
    Inspect the model structure to understand FFN organization.
    """
    print(f"Loading model from {model_path}...")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    print("\n" + "=" * 80)
    print("Model Structure Inspection")
    print("=" * 80)

    # Access language model
    language_model = model.language_model

    print(f"\nLanguage model type: {type(language_model)}")
    print(f"Number of layers: {len(language_model.model.layers)}")

    # Inspect first layer
    print("\n" + "-" * 80)
    print("First Layer Structure:")
    print("-" * 80)
    layer0 = language_model.model.layers[0]
    print(f"Layer type: {type(layer0)}")
    print(f"\nLayer components:")
    for name, module in layer0.named_children():
        print(f"  - {name}: {type(module)}")

    # Inspect MLP structure
    print("\n" + "-" * 80)
    print("MLP (FFN) Structure:")
    print("-" * 80)
    mlp = layer0.mlp
    print(f"MLP type: {type(mlp)}")
    print(f"\nMLP components:")
    for name, module in mlp.named_children():
        print(f"  - {name}: {type(module)}")
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            print(f"    Shape: {module.in_features} -> {module.out_features}")

    # Print all layer names
    print("\n" + "-" * 80)
    print("All Language Model Layers:")
    print("-" * 80)
    for idx, layer in enumerate(language_model.model.layers):
        print(f"Layer {idx}: {type(layer.mlp)}")
        if idx == 0:
            print(f"  gate_proj: {layer.mlp.gate_proj.in_features} -> {layer.mlp.gate_proj.out_features}")
            print(f"  up_proj: {layer.mlp.up_proj.in_features} -> {layer.mlp.up_proj.out_features}")
            print(f"  down_proj: {layer.mlp.down_proj.in_features} -> {layer.mlp.down_proj.out_features}")

    print("\n" + "=" * 80)
    print("Total trainable parameters:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} ({total_params / 1e9:.2f}B)")
    print("=" * 80)


def main():
    model_path = "/gpfs/volcano/models/llava-v1.5-7b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("LLaVA Model Structure Test")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print("=" * 80)

    try:
        inspect_model_structure(model_path, device)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
