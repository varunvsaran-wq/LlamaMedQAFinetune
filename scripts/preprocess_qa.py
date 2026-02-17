"""Convert MedQA train.jsonl to LLaMA-Factory Alpaca SFT format."""

import json
from pathlib import Path

INPUT_PATH = Path("data_clean/data_clean/questions/US/train.jsonl")
OUTPUT_PATH = Path("data/medqa_sft.jsonl")

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def format_example(record: dict) -> dict:
    question = record["question"]
    options = record["options"]
    answer_idx = record["answer_idx"]
    answer_text = record["answer"]

    # Build options string
    options_str = "\n".join(
        f"{letter}. {options[letter]}" for letter in OPTION_LETTERS if letter in options
    )

    instruction = (
        "You are a medical expert. Answer the following USMLE-style question "
        "by selecting the correct option.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}"
    )

    output = f"The correct answer is {answer_idx}. {answer_text}"

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line.strip())
            formatted = format_example(record)
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
