"""
Evaluate baseline (unpruned) LLaVA model on OCR-VQA dataset.
This script serves as a baseline for comparison with pruned models.
Uses ROUGE metric like FastV.
"""

import os
import json
from datetime import datetime
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

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


def evaluate_ocrvqa(model, tokenizer, image_processor, dataset, num_eval_samples, device):
    """
    Evaluate model on OCR-VQA dataset (single sample loop like FastV).
    """
    print("=" * 80)
    print(f"Evaluating on OCR-VQA ({num_eval_samples} samples, greedy decoding)")
    print("=" * 80)

    # Determine conversation mode
    model_name = model.config._name_or_path
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"

    # Args for process_images
    class Args:
        image_aspect_ratio = "pad"

    args = Args()

    model.eval()
    predictions = []
    labels = []

    skipped = 0

    for idx in tqdm(range(num_eval_samples), desc="Evaluating"):
        example = dataset[idx]

        # Image processing
        try:
            image = example["image"]
            # Skip tiny images
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

        # Get question and answer
        question = example.get("question", example.get("questions", [""])[0])
        answer = example.get("answer", example.get("answers", [""])[0])

        # Create conversation
        conv = conv_templates[conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
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
            print(f"  Q:  {question}")
            print(f"  GT: {answer}")
            print(f"  Pred: '{output}'")

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate baseline LLaVA model on OCR-VQA"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/gpfs/volcano/models/llava-v1.5-7b",
        help="Path to LLaVA model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/gpfs/volcano/models/howard-hou-OCR-VQA",
        help="Path to OCR-VQA dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=None,
        help="Number of samples for evaluation (None = use all test data)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on"
    )

    args = parser.parse_args()

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"baseline_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Baseline LLaVA Model Evaluation on OCR-VQA")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Evaluation samples: {args.num_eval_samples}")
    print("=" * 80)

    # 加载模型
    print("\nLoading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=args.device,
    )
    print(f"Model loaded: {model_name}")
    print(f"Context length: {context_len}")

    # 加载数据集（test split）
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset_path)["test"]
    print(f"Dataset loaded: {len(dataset)} total samples")

    # 确定评估样本数
    if args.num_eval_samples is None:
        num_eval_samples = len(dataset)
        print(f"Will evaluate on ALL {num_eval_samples} test samples")
    else:
        num_eval_samples = min(args.num_eval_samples, len(dataset))
        print(f"Will evaluate on {num_eval_samples} samples")

    # 评估
    predictions, labels = evaluate_ocrvqa(
        model, tokenizer, image_processor, dataset, num_eval_samples, args.device
    )

    # 计算ROUGE指标（like FastV）
    rouge_score, rouge_scores = compute_rouge(labels, predictions)

    print("\n" + "=" * 80)
    print("Evaluation Metrics:")
    print("=" * 80)
    print(f"Total samples: {len(predictions)}")
    print(f"ROUGE score: {rouge_score:.4f}")
    print("=" * 80)

    # 保存详细结果
    results_dict = {
        "predictions": predictions,
        "labels": labels,
        "rouge_scores": rouge_scores.tolist()
    }
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\nEvaluation results saved to {eval_path}")

    # 保存 summary
    summary = {
        "model": args.model_path,
        "dataset": args.dataset_path,
        "num_eval_samples": len(predictions),
        "rouge_score": float(rouge_score),
        "timestamp": timestamp,
        "model_type": "baseline (unpruned)",
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()