"""
Prune LLaVA model using Wanda method on LLM Transformer layers.
This script applies Wanda pruning to the LLaMA decoder layers of LLaVA model,
evaluates on OCR-VQA dataset, and saves intermediate results.
Uses ROUGE metric like FastV.
"""

import torch
import torch.nn as nn
from datasets import load_dataset
import os
import numpy as np
import json
from tqdm import tqdm
import argparse
from datetime import datetime
from PIL import Image
import re

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# ============ Text normalization (from FastV) ============
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


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.
    Copied from Wanda implementation.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of LLM layers in the model.
    """
    layers = model.model.layers
    count = 0
    total_params = 0

    layer_sparsity = {}

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        layer_sparsity[f'layer_{i}'] = float(sub_count)/sub_params
        print(f"Layer {i} sparsity: {layer_sparsity[f'layer_{i}']:.6f}")

    overall_sparsity = float(count)/total_params
    return overall_sparsity, layer_sparsity


class WrappedGPT:
    """
    Wrapper for capturing activation statistics.
    Simplified from Wanda's WrappedGPT.
    """
    def __init__(self, layer):
        self.layer = layer
        self.scaler_row = None
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if self.scaler_row is None:
            self.scaler_row = torch.zeros(inp.shape[0], device=inp.device, dtype=inp.dtype)

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


def prepare_calibration_input(model, dataset, tokenizer, image_processor, nsamples, device):
    """
    Prepare calibration data from OCR-VQA dataset for pruning.
    """
    print(f"Preparing {nsamples} calibration samples from OCR-VQA...")

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

    # Collect samples
    calibration_data = []
    for idx in range(nsamples):
        example = dataset[idx]
        image = example['image']
        # Force robust RGB conversion for ALL images
        # Convert to numpy first to handle all PIL formats properly
        img_array = np.array(image)
        # Ensure 3 channels
        if len(img_array.shape) == 2:
            # Grayscale: replicate to 3 channels
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            # Single channel with explicit dimension: replicate
            img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA: drop alpha channel
            img_array = img_array[:, :, :3]
        # Create new RGB PIL image from numpy array
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

        calibration_data.append({
            'input_ids': input_ids,
            'images': image_tensor,
            'question': question
        })

    print(f"Calibration data prepared: {len(calibration_data)} samples")
    return calibration_data


def prune_wanda_vlm(model, tokenizer, image_processor, dataset, nsamples, sparsity_ratio, device):
    """
    Apply Wanda pruning to LLM layers of VLM model.

    Args:
        model: LLaVA model
        tokenizer: Tokenizer
        image_processor: Image processor
        dataset: OCR-VQA dataset
        nsamples: Number of calibration samples
        sparsity_ratio: Target sparsity (0-1)
        device: Device to run on

    Returns:
        Dictionary with pruning statistics
    """
    print("="*80)
    print("Starting Wanda Pruning on LLM Transformer Layers")
    print("="*80)

    # Prepare calibration data
    calibration_data = prepare_calibration_input(
        model, dataset, tokenizer, image_processor, nsamples, device
    )

    # Get LLM layers
    layers = model.model.layers
    num_layers = len(layers)
    print(f"Total LLM layers: {num_layers}")

    # Storage for pruning statistics
    pruning_stats = {
        'num_layers': num_layers,
        'sparsity_ratio': sparsity_ratio,
        'nsamples': nsamples,
        'layer_stats': {}
    }

    model.eval()

    # For each layer, compute activations and prune
    for layer_idx in tqdm(range(num_layers), desc="Pruning layers"):
        layer = layers[layer_idx]
        subset = find_layers(layer)

        # Wrap layers to capture activations
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def hook_fn(module, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return hook_fn

        # Register hooks
        handles = []
        for name in wrapped_layers:
            wrapped_layers[name].nsamples = nsamples
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Forward pass to collect activation statistics
        for sample in calibration_data:
            with torch.no_grad():
                _ = model(
                    input_ids=sample['input_ids'],
                    images=sample['images'],
                    return_dict=True
                )

        # Remove hooks
        for h in handles:
            h.remove()

        # Prune each Linear layer in this Transformer layer
        layer_stats = {}
        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # Compute threshold for unstructured pruning
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)].cpu()
            W_mask = (W_metric <= thresh)

            # Apply pruning
            W[W_mask] = 0

            # Record statistics
            pruned_params = (W==0).sum().item()
            total_params = W.numel()
            actual_sparsity = pruned_params / total_params

            layer_stats[name] = {
                'pruned_params': int(pruned_params),
                'total_params': int(total_params),
                'actual_sparsity': float(actual_sparsity),
                'target_sparsity': float(sparsity_ratio)
            }

            print(f"  Layer {layer_idx} {name}: {actual_sparsity:.4f} sparsity ({pruned_params}/{total_params})")

        pruning_stats['layer_stats'][f'layer_{layer_idx}'] = layer_stats

    print("\nPruning completed!")
    return pruning_stats


def evaluate_ocrvqa(model, tokenizer, image_processor, dataset, num_eval_samples, device):
    """
    Evaluate model on OCR-VQA dataset (single sample loop like FastV).
    Returns predictions and labels for ROUGE computation.
    """
    print("="*80)
    print(f"Evaluating on OCR-VQA ({num_eval_samples} samples, greedy decoding)")
    print("="*80)

    # Determine conversation mode
    model_name = model.config._name_or_path
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    # Args for process_images
    class Args:
        image_aspect_ratio = 'pad'
    args = Args()

    predictions = []
    labels = []
    skipped = 0
    model.eval()

    for idx in tqdm(range(num_eval_samples), desc="Evaluating"):
        example = dataset[idx]

        # Image processing
        try:
            image = example['image']
            # Skip tiny images
            if image.size[0] < 2 or image.size[1] < 2:
                skipped += 1
                continue
            image = image.convert('RGB')
            image_tensor = process_images([image], image_processor, args)
            image_tensor = image_tensor.to(device, dtype=torch.float16)
        except Exception as e:
            print(f"\n[WARNING] Skipping sample {idx} due to image error: {e}")
            skipped += 1
            continue

        # Get question and answer
        question = example.get('question', example.get('questions', [''])[0])
        answer = example.get('answer', example.get('answers', [''])[0])

        # Create conversation
        conv = conv_templates[conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(device)

        # Generate (greedy decoding, like FastV)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=50,
                use_cache=True,
            )

        # Decode (output_ids only contains generated tokens, not input)
        output = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip().replace("</s>", "")

        predictions.append(output)
        labels.append([answer])

        # Print first few examples
        if idx < 5:
            print(f"\nExample {idx}:")
            print(f"  Q: {question}")
            print(f"  GT: {answer}")
            print(f"  Pred: {output}")

    print(f"\nSkipped {skipped} samples due to errors")
    return predictions, labels


def compute_rouge(labels, predictions):
    """
    Compute ROUGE score like FastV.
    """
    from pycocoevalcap.rouge.rouge import Rouge

    scorer = Rouge()
    labels_dict = {}
    predictions_dict = {}

    for i in range(len(labels)):
        labels_dict[i] = [clean_text(t) for t in labels[i]]
        predictions_dict[i] = [clean_text(predictions[i])]

    (score, scores) = scorer.compute_score(labels_dict, predictions_dict)
    return score, scores


def main():
    parser = argparse.ArgumentParser(description='Prune LLaVA model using Wanda method')
    parser.add_argument('--model_path', type=str, default='/gpfs/volcano/models/llava-v1.5-7b',
                        help='Path to LLaVA model')
    parser.add_argument('--dataset_path', type=str, default='/gpfs/volcano/models/howard-hou-OCR-VQA',
                        help='Path to OCR-VQA dataset')
    parser.add_argument('--output_dir', type=str, default='./pruning_results',
                        help='Directory to save results')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples for pruning')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5,
                        help='Target sparsity ratio (0-1)')
    parser.add_argument('--num_eval_samples', type=int, default=None,
                        help='Number of samples for evaluation (None = use all test data)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save pruned model (optional)')

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'wanda_prune_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("VLM Pruning with Wanda Method")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Sparsity ratio: {args.sparsity_ratio}")
    print(f"Evaluation samples: {args.num_eval_samples}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=args.device
    )
    print(f"Model loaded: {model_name}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset_path)["test"]
    print(f"Dataset loaded: {len(dataset)} samples")

    # Check sparsity before pruning
    print("\n" + "="*80)
    print("Sparsity BEFORE pruning:")
    print("="*80)
    sparsity_before, layer_sparsity_before = check_sparsity(model)
    print(f"Overall sparsity: {sparsity_before:.6f}")

    # Apply Wanda pruning
    pruning_stats = prune_wanda_vlm(
        model, tokenizer, image_processor, dataset,
        args.nsamples, args.sparsity_ratio, args.device
    )

    # Check sparsity after pruning
    print("\n" + "="*80)
    print("Sparsity AFTER pruning:")
    print("="*80)
    sparsity_after, layer_sparsity_after = check_sparsity(model)
    print(f"Overall sparsity: {sparsity_after:.6f}")

    # Save pruning statistics
    pruning_stats['sparsity_before'] = sparsity_before
    pruning_stats['sparsity_after'] = sparsity_after
    pruning_stats['layer_sparsity_before'] = layer_sparsity_before
    pruning_stats['layer_sparsity_after'] = layer_sparsity_after

    stats_path = os.path.join(output_dir, 'pruning_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(pruning_stats, f, indent=2)
    print(f"\nPruning statistics saved to {stats_path}")

    # Use all test data if num_eval_samples is None
    if args.num_eval_samples is None:
        num_eval_samples = len(dataset)
        print(f"\nWill evaluate on ALL {num_eval_samples} test samples")
    else:
        num_eval_samples = min(args.num_eval_samples, len(dataset))
        print(f"\nWill evaluate on {num_eval_samples} samples")

    # Evaluate on OCR-VQA
    predictions, labels = evaluate_ocrvqa(
        model, tokenizer, image_processor, dataset,
        num_eval_samples, args.device
    )

    # Compute ROUGE score (like FastV)
    rouge_score, rouge_scores = compute_rouge(labels, predictions)

    # Save evaluation results
    results_dict = {
        "predictions": predictions,
        "labels": labels,
        "rouge_scores": rouge_scores.tolist()
    }
    eval_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to {eval_path}")

    # Save summary
    summary = {
        'model': args.model_path,
        'sparsity_ratio': args.sparsity_ratio,
        'sparsity_before': sparsity_before,
        'sparsity_after': sparsity_after,
        'nsamples': args.nsamples,
        'num_eval_samples': len(predictions),
        'rouge_score': float(rouge_score),
        'timestamp': timestamp,
        'model_type': 'pruned (wanda)',
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Sparsity before: {sparsity_before:.6f}")
    print(f"Sparsity after: {sparsity_after:.6f}")
    print(f"Total samples: {len(predictions)}")
    print(f"ROUGE score: {rouge_score:.4f}")
    print("="*80)

    # Optionally save pruned model
    if args.save_model:
        print(f"\nSaving pruned model to {args.save_model}...")
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("Model saved!")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()