"""
MULTIFLOW Pruning and Evaluation for LLaVA

This script implements the complete MULTIFLOW pipeline:
1. Step 1: Compute adaptive sparsity distribution (modality-aware)
2. Step 2: Compute information flow scores with activation statistics
3. Step 3: Apply pruning using Step 1 distribution + Step 2 scores
4. Evaluate on OCR-VQA dataset with ROUGE metric

Usage:
    python multiflow_prune_and_eval.py --target_sparsity 0.5 --nsamples 128 --num_eval_samples 5000
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from PIL import Image
from datetime import datetime

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# For loading calibration data
from datasets import load_dataset


# ============================================================================
# Text normalization (from FastV/baseline code)
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
    """
    Compute ROUGE score like FastV/baseline.
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


# ============================================================================
# MULTIFLOW Implementation (from baseline/multiflow_step1_step2_compute_scores.py)
# ============================================================================

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
    """
    if 'vision_tower' in param_name:
        return 'vision'
    elif 'mm_projector' in param_name:
        return 'fusion'
    elif 'model.layers' in param_name:
        return 'fusion'
    elif 'embed_tokens' in param_name or 'lm_head' in param_name:
        return 'text'
    else:
        return 'fusion'


def compute_multimodal_distribution(model, target_sparsity=0.5):
    """
    Step 1 of MULTIFLOW: Compute adaptive sparsity distribution.
    """
    print("="*80)
    print("MULTIFLOW Step 1: Computing Adaptive Sparsity Distribution")
    print("="*80)
    print(f"Target global sparsity: {target_sparsity*100:.1f}%\n")

    linear_layers = find_linear_layers(model)
    print(f"Found {len(linear_layers)} Linear layers in the model\n")

    modality_layers = defaultdict(dict)
    modality_scores = defaultdict(list)

    for name, layer in linear_layers.items():
        modality = detect_modality_llava(name)
        W = layer.weight.data
        score = torch.abs(W)

        modality_layers[modality][name] = {
            'weight': W,
            'score': score,
            'numel': W.numel()
        }
        modality_scores[modality].append(score.flatten())

    layer_distribution = {}
    modality_stats = {}

    for modality in sorted(modality_layers.keys()):
        all_scores = torch.cat(modality_scores[modality])
        total_elements = all_scores.numel()

        k = int(total_elements * target_sparsity)
        threshold, _ = torch.kthvalue(all_scores.cpu(), k=k)

        modality_total_params = 0
        modality_total_pruned = 0
        layer_sparsities = []

        for layer_name, layer_info in modality_layers[modality].items():
            score = layer_info['score']
            mask = (score > threshold)

            pruned_params = (~mask).sum().item()
            total_params_layer = score.numel()
            actual_sparsity = pruned_params / total_params_layer

            layer_distribution[layer_name] = actual_sparsity
            layer_sparsities.append(actual_sparsity)

            modality_total_params += total_params_layer
            modality_total_pruned += pruned_params

        modality_actual_sparsity = modality_total_pruned / modality_total_params
        modality_stats[modality] = {
            'num_layers': len(modality_layers[modality]),
            'total_params': modality_total_params,
            'pruned_params': modality_total_pruned,
            'actual_sparsity': modality_actual_sparsity,
        }

    print(f"✓ Step 1 completed\n")
    return layer_distribution, modality_stats


def prepare_calibration_data_llava(model, tokenizer, image_processor, dataset, nsamples, device):
    """
    Prepare calibration data from OCR-VQA dataset.
    """
    print(f"Preparing {nsamples} calibration samples...")

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

        question = example.get('question', example.get('questions', [""])[0])

        conv = conv_templates[conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(device, dtype=torch.float16)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)

        calibration_data.append({
            'input_ids': input_ids,
            'images': image_tensor,
            'attention_mask': attention_mask,
        })

    print(f"✓ Calibration data prepared: {len(calibration_data)} samples")
    return calibration_data


class ActivationCollector:
    """Hook-based activation collector for MULTIFLOW."""

    def __init__(self, layer, layer_name, modality):
        self.layer = layer
        self.layer_name = layer_name
        self.modality = modality
        self.input_history = []

    def hook(self, module, input, output):
        inp = input[0].data
        self.input_history.append(inp.clone().detach())

    def clear(self):
        self.input_history = []


def compute_activation_norms(linear_layers, calibration_data, model, device):
    """
    Step 2.1-2.3: Collect activation statistics via forward passes.
    """
    print("\n" + "="*80)
    print("MULTIFLOW Step 2: Computing Activation Statistics")
    print("="*80)
    print(f"Collecting activations from {len(calibration_data)} samples...\n")

    collectors = {}
    handles = []

    for layer_name, layer in linear_layers.items():
        modality = detect_modality_llava(layer_name)
        collector = ActivationCollector(layer, layer_name, modality)
        collectors[layer_name] = collector
        handle = layer.register_forward_hook(collector.hook)
        handles.append(handle)

    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(calibration_data, desc="Collecting activations")):
            _ = model(
                input_ids=sample['input_ids'],
                images=sample['images'],
                return_dict=True
            )
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()

    print("\n✓ Activation collection completed!")
    print("Computing activation norms for each layer...")

    activation_norms = {}

    for layer_name, collector in tqdm(collectors.items(), desc="Computing norms"):
        if len(collector.input_history) == 0:
            W = linear_layers[layer_name].weight.data
            activation_norms[layer_name] = torch.ones(W.shape[1], dtype=torch.float32)
            continue

        all_inputs = []
        for inp in collector.input_history:
            if len(inp.shape) == 3:
                inp_reshaped = inp.reshape(-1, inp.shape[-1])
            elif len(inp.shape) == 2:
                inp_reshaped = inp
            else:
                continue
            all_inputs.append(inp_reshaped)

        if len(all_inputs) == 0:
            W = linear_layers[layer_name].weight.data
            activation_norms[layer_name] = torch.ones(W.shape[1], dtype=torch.float32)
            continue

        X = torch.cat(all_inputs, dim=0)
        nsamples = X.shape[0]
        actn_norm = torch.norm(X.cpu().float(), p=2, dim=0) ** 2 / nsamples

        activation_norms[layer_name] = actn_norm

        collector.clear()
        del X
        torch.cuda.empty_cache()

    print(f"\n✓ Activation norms computed for {len(activation_norms)} layers")
    return activation_norms


def compute_information_flow_scores(linear_layers, activation_norms):
    """
    Step 2.4-2.5: Compute MULTIFLOW's information flow scores.

    Formula: Score = Imp(output) ⊗ Imp(input) ⊙ |W|
    """
    print("\n" + "="*80)
    print("Computing Information Flow Scores (MULTIFLOW Formula)")
    print("="*80)
    print("\nFormula: Score = Imp(output) ⊗ Imp(input) ⊙ |W|")
    print("="*80 + "\n")

    information_flow_scores = {}

    for layer_name, layer in tqdm(linear_layers.items(), desc="Computing scores"):
        W = layer.weight.data
        actn_norm = torch.sqrt(activation_norms[layer_name]).to(W.device)

        importance_per_output = (W.abs() * actn_norm).mean(dim=1)
        importance_per_input = (W.abs() * actn_norm).mean(dim=0)

        score = torch.outer(importance_per_output, importance_per_input)
        final_score = score * W.abs()

        information_flow_scores[layer_name] = final_score.cpu()

    print(f"\n✓ Information flow scores computed for {len(information_flow_scores)} layers")
    return information_flow_scores


def apply_multiflow_pruning(linear_layers, layer_distribution, information_flow_scores):
    """
    Step 3: Apply pruning using distribution from Step 1 and scores from Step 2.
    """
    print("\n" + "="*80)
    print("MULTIFLOW Step 3: Applying Pruning")
    print("="*80)

    total_params = 0
    total_pruned = 0
    layer_stats = {}

    for layer_name, layer in tqdm(linear_layers.items(), desc="Pruning layers"):
        target_sparsity = layer_distribution[layer_name]
        scores = information_flow_scores[layer_name].to(layer.weight.device)

        W = layer.weight.data
        k = int(W.numel() * target_sparsity)

        if k == 0:
            layer_stats[layer_name] = {
                'target_sparsity': 0.0,
                'actual_sparsity': 0.0,
                'total_params': W.numel(),
                'pruned_params': 0
            }
            total_params += W.numel()
            continue

        threshold = torch.kthvalue(scores.flatten(), k)[0]
        mask = (scores > threshold)
        W[~mask] = 0

        actual_pruned = (~mask).sum().item()
        actual_sparsity = actual_pruned / W.numel()

        layer_stats[layer_name] = {
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'total_params': W.numel(),
            'pruned_params': actual_pruned
        }

        total_params += W.numel()
        total_pruned += actual_pruned

    overall_sparsity = total_pruned / total_params if total_params > 0 else 0

    print(f"\n✓ Pruning completed!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Pruned parameters: {total_pruned:,}")
    print(f"  Overall sparsity: {overall_sparsity*100:.2f}%")

    return {
        'total_params': total_params,
        'total_pruned': total_pruned,
        'overall_sparsity': overall_sparsity
    }


def evaluate_ocrvqa(model, tokenizer, image_processor, dataset, num_eval_samples, device):
    """
    Evaluate model on OCR-VQA dataset (same as baseline).
    """
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

        # Greedy decoding (same as baseline)
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
        description='MULTIFLOW Pruning and Evaluation for LLaVA',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model_path', type=str,
                        default='/gpfs/volcano/models/llava-v1.5-7b',
                        help='Path to LLaVA model')
    parser.add_argument('--target_sparsity', type=float, default=0.5,
                        help='Target global sparsity ratio (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to load model on')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples for pruning')
    parser.add_argument('--dataset_path', type=str,
                        default='/gpfs/volcano/models/howard-hou-OCR-VQA',
                        help='Path to OCR-VQA dataset')
    parser.add_argument('--num_eval_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--output_dir', type=str, default='./multiflow_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"multiflow_sp{args.target_sparsity}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("MULTIFLOW Pruning and Evaluation for LLaVA".center(80))
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Target sparsity: {args.target_sparsity*100:.1f}%")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Evaluation samples: {args.num_eval_samples}")
    print(f"Output directory: {output_dir}")
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
    print(f"✓ Model loaded: {model_name}\n")

    # Get all linear layers
    linear_layers = find_linear_layers(model)
    print(f"Found {len(linear_layers)} Linear layers")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset_path)["test"]
    print(f"✓ Dataset loaded: {len(dataset)} total samples")

    # ========================================================================
    # Step 1: Compute adaptive sparsity distribution
    # ========================================================================
    layer_distribution, modality_stats = compute_multimodal_distribution(
        model, target_sparsity=args.target_sparsity
    )

    # ========================================================================
    # Step 2: Compute information flow scores
    # ========================================================================
    calibration_data = prepare_calibration_data_llava(
        model, tokenizer, image_processor, dataset, args.nsamples, args.device
    )

    activation_norms = compute_activation_norms(
        linear_layers, calibration_data, model, args.device
    )

    information_flow_scores = compute_information_flow_scores(
        linear_layers, activation_norms
    )

    # ========================================================================
    # Step 3: Apply pruning
    # ========================================================================
    pruning_stats = apply_multiflow_pruning(
        linear_layers, layer_distribution, information_flow_scores
    )

    # ========================================================================
    # Evaluate on OCR-VQA
    # ========================================================================
    if args.num_eval_samples is None:
        num_eval_samples = len(dataset)
    else:
        num_eval_samples = min(args.num_eval_samples, len(dataset))

    predictions, labels = evaluate_ocrvqa(
        model, tokenizer, image_processor, dataset, num_eval_samples, args.device
    )

    # Compute ROUGE
    rouge_score, rouge_scores = compute_rouge(labels, predictions)

    print("\n" + "="*80)
    print("Evaluation Results:")
    print("="*80)
    print(f"Total samples evaluated: {len(predictions)}")
    print(f"Overall sparsity: {pruning_stats['overall_sparsity']*100:.2f}%")
    print(f"ROUGE score: {rouge_score:.4f}")
    print("="*80)

    # Save results
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
        "method": "MULTIFLOW",
        "target_sparsity": args.target_sparsity,
        "actual_sparsity": pruning_stats['overall_sparsity'],
        "num_calibration_samples": args.nsamples,
        "num_eval_samples": len(predictions),
        "rouge_score": float(rouge_score),
        "timestamp": timestamp,
        "modality_stats": modality_stats,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - {eval_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
