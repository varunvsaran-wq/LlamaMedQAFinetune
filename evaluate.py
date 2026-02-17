"""Evaluate fine-tuned Llama 3 on MedQA: perplexity and accuracy."""

import json
import re
import math
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
ADAPTER_PATH = "saves/llama3-8b-medqa/lora/sft"
DEV_PATH = Path("data_clean/data_clean/questions/US/dev.jsonl")
TEST_PATH = Path("data_clean/data_clean/questions/US/test.jsonl")

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def load_model(adapter_path: str):
    """Load base model in 4-bit with LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")
    else:
        print(f"WARNING: No adapter found at {adapter_path}, using base model")

    model.eval()
    return model, tokenizer


def format_question(record: dict) -> str:
    """Format a question for the model."""
    options = record["options"]
    options_str = "\n".join(
        f"{letter}. {options[letter]}" for letter in OPTION_LETTERS if letter in options
    )
    return (
        "You are a medical expert. Answer the following USMLE-style question "
        "by selecting the correct option.\n\n"
        f"Question: {record['question']}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer:"
    )


def compute_perplexity(model, tokenizer, data_path: Path, max_samples: int = 500) -> float:
    """Compute perplexity on formatted QA pairs."""
    total_loss = 0.0
    total_tokens = 0

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:max_samples]

    for line in lines:
        record = json.loads(line.strip())
        prompt = format_question(record)
        answer = f" The correct answer is {record['answer_idx']}. {record['answer']}"
        full_text = prompt + answer

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def evaluate_accuracy(model, tokenizer, data_path: Path, max_samples: int = 500) -> float:
    """Evaluate answer accuracy by generating responses and extracting the answer letter."""
    correct = 0
    total = 0

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[:max_samples]

    for line in lines:
        record = json.loads(line.strip())
        prompt = format_question(record)
        gold = record["answer_idx"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Extract predicted answer letter
        predicted = extract_answer(generated)
        if predicted == gold:
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  Progress: {total}/{len(lines)}, running accuracy: {correct/total:.3f}")

    return correct / total if total > 0 else 0.0


def extract_answer(text: str) -> str:
    """Extract the answer letter from model output."""
    # Match patterns like "The correct answer is E" or just a standalone letter
    match = re.search(r"correct answer is\s+([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: first standalone letter A-E
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)

    return ""


def main():
    print("Loading model...")
    model, tokenizer = load_model(ADAPTER_PATH)

    # Perplexity on dev set
    if DEV_PATH.exists():
        print("\nComputing perplexity on dev set...")
        ppl = compute_perplexity(model, tokenizer, DEV_PATH)
        print(f"Dev set perplexity: {ppl:.2f}")

    # Accuracy on test set
    if TEST_PATH.exists():
        print("\nEvaluating accuracy on test set...")
        acc = evaluate_accuracy(model, tokenizer, TEST_PATH)
        print(f"Test set accuracy: {acc:.3f} ({acc*100:.1f}%)")


if __name__ == "__main__":
    main()
