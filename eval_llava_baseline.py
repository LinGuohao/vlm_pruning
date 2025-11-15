"""
Evaluate baseline (unpruned) LLaVA model on OCR-VQA dataset.
This script serves as a baseline for comparison with pruned models.
"""

import torch
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import argparse
from datetime import datetime

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def evaluate_ocrvqa(model, tokenizer, image_processor, dataset, num_eval_samples, device, batch_size=8):
    """
    Evaluate model on OCR-VQA dataset with batch processing.
    Returns generated answers for comparison.
    """
    print("="*80)
    print(f"Evaluating on OCR-VQA ({num_eval_samples} samples, batch_size={batch_size})")
    print("="*80)

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

    results = []
    model.eval()

    # Process in batches
    num_batches = (num_eval_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_eval_samples)
        batch_examples = [dataset[i] for i in range(start_idx, end_idx)]

        batch_images = []
        batch_input_ids = []
        batch_questions = []
        batch_answers = []

        for example in batch_examples:
            image = example['image'].convert('RGB')

            # Get question and answer
            if 'question' in example:
                question = example['question']
            elif 'questions' in example:
                question = example['questions'][0]
            else:
                question = ""

            if 'answer' in example:
                answer = example['answer']
            elif 'answers' in example:
                answer = example['answers'][0]
            else:
                answer = ""

            # Create conversation
            conv = conv_templates[conv_mode].copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Process image
            image_tensor = process_images([image], image_processor, args)
            batch_images.append(image_tensor)

            # Tokenize input
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            batch_input_ids.append(input_ids)

            batch_questions.append(question)
            batch_answers.append(answer)

        # Stack images
        batch_images = torch.cat(batch_images, dim=0).to(device, dtype=torch.float16)

        # Pad input_ids to same length
        max_len = max(ids.shape[0] for ids in batch_input_ids)
        padded_input_ids = []
        attention_mask = []
        for ids in batch_input_ids:
            padding_length = max_len - ids.shape[0]
            padded_ids = torch.cat([
                torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype),
                ids
            ])
            padded_input_ids.append(padded_ids)
            mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long),
                torch.ones(ids.shape[0], dtype=torch.long)
            ])
            attention_mask.append(mask)

        batch_input_ids = torch.stack(padded_input_ids).to(device)
        attention_mask = torch.stack(attention_mask).to(device)

        # Generate (greedy decoding for deterministic results)
        with torch.inference_mode():
            output_ids = model.generate(
                batch_input_ids,
                images=batch_images,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=50,
                use_cache=False,  # Disable KV cache to avoid cache_position compatibility issues
            )

        # Decode outputs
        for i, output_id in enumerate(output_ids):
            # Find the start of generated text (after input)
            input_len = batch_input_ids[i].shape[0]
            output = tokenizer.decode(output_id[input_len:], skip_special_tokens=True).strip()

            sample_idx = start_idx + i
            results.append({
                'sample_id': sample_idx,
                'question': batch_questions[i],
                'ground_truth': batch_answers[i],
                'prediction': output
            })

            # Print first few examples
            if sample_idx < 5:
                print(f"\nExample {sample_idx}:")
                print(f"  Q: {batch_questions[i]}")
                print(f"  GT: {batch_answers[i]}")
                print(f"  Pred: {output}")

    return results


def compute_metrics(results):
    """
    Compute evaluation metrics.
    """
    total = len(results)

    # Exact match (case-insensitive)
    exact_match = sum(1 for r in results if r['prediction'].lower().strip() == r['ground_truth'].lower().strip())
    exact_match_acc = exact_match / total if total > 0 else 0

    # Partial match (prediction contains ground truth or vice versa)
    partial_match = sum(1 for r in results
                       if r['ground_truth'].lower() in r['prediction'].lower()
                       or r['prediction'].lower() in r['ground_truth'].lower())
    partial_match_acc = partial_match / total if total > 0 else 0

    metrics = {
        'total_samples': total,
        'exact_match': exact_match,
        'exact_match_accuracy': exact_match_acc,
        'partial_match': partial_match,
        'partial_match_accuracy': partial_match_acc
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline LLaVA model on OCR-VQA')
    parser.add_argument('--model_path', type=str, default='/gpfs/volcano/models/llava-v1.5-7b',
                        help='Path to LLaVA model')
    parser.add_argument('--dataset_path', type=str, default='/gpfs/volcano/models/howard-hou-OCR-VQA',
                        help='Path to OCR-VQA dataset')
    parser.add_argument('--output_dir', type=str, default='./baseline_results',
                        help='Directory to save results')
    parser.add_argument('--num_eval_samples', type=int, default=None,
                        help='Number of samples for evaluation (None = use all test data)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'baseline_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Baseline LLaVA Model Evaluation on OCR-VQA")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {output_dir}")
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
    print(f"Context length: {context_len}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset_path)["test"]
    print(f"Dataset loaded: {len(dataset)} total samples")

    # Use all test data if num_eval_samples is None
    if args.num_eval_samples is None:
        num_eval_samples = len(dataset)
        print(f"Will evaluate on ALL {num_eval_samples} test samples")
    else:
        num_eval_samples = min(args.num_eval_samples, len(dataset))
        print(f"Will evaluate on {num_eval_samples} samples")

    # Evaluate on OCR-VQA
    eval_results = evaluate_ocrvqa(
        model, tokenizer, image_processor, dataset,
        num_eval_samples, args.device, args.batch_size
    )

    # Compute metrics
    metrics = compute_metrics(eval_results)

    print("\n" + "="*80)
    print("Evaluation Metrics:")
    print("="*80)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Exact match: {metrics['exact_match']} ({metrics['exact_match_accuracy']:.4f})")
    print(f"Partial match: {metrics['partial_match']} ({metrics['partial_match_accuracy']:.4f})")
    print("="*80)

    # Save evaluation results
    eval_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to {eval_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save summary
    summary = {
        'model': args.model_path,
        'dataset': args.dataset_path,
        'num_eval_samples': num_eval_samples,
        'metrics': metrics,
        'timestamp': timestamp,
        'model_type': 'baseline (unpruned)'
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Create a simple report
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Baseline LLaVA Model Evaluation Report\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.dataset_path}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Evaluation samples: {num_eval_samples}\n")
        f.write("\n")
        f.write("Metrics:\n")
        f.write(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match']}/{metrics['total_samples']})\n")
        f.write(f"  Partial Match Accuracy: {metrics['partial_match_accuracy']:.4f} ({metrics['partial_match']}/{metrics['total_samples']})\n")
        f.write("="*80 + "\n")
    print(f"Report saved to {report_path}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()