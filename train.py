"""Orchestrator: preprocess data and run two-stage training."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    project_root = Path(__file__).parent

    # Step 1: Preprocess data if needed
    sft_data = project_root / "data" / "medqa_sft.jsonl"
    pretrain_data = project_root / "data" / "textbooks_pretrain.jsonl"

    if not sft_data.exists():
        run_command(
            [sys.executable, "scripts/preprocess_qa.py"],
            "Preprocessing QA data -> Alpaca SFT format",
        )
    else:
        print(f"SFT data already exists: {sft_data}")

    if not pretrain_data.exists():
        run_command(
            [sys.executable, "scripts/preprocess_textbooks.py"],
            "Preprocessing textbooks -> pretraining chunks",
        )
    else:
        print(f"Pretrain data already exists: {pretrain_data}")

    # Step 2: Stage 1 - Continued pretraining on textbooks
    stage1_output = project_root / "saves" / "llama3-8b-medqa" / "lora" / "pretrain"
    if not (stage1_output / "adapter_model.safetensors").exists():
        run_command(
            ["llamafactory-cli", "train", "configs/stage1_pretrain.yaml"],
            "Stage 1: Continued Pretraining on Medical Textbooks",
        )
    else:
        print(f"Stage 1 adapter already exists: {stage1_output}")

    # Step 3: Stage 2 - SFT on QA pairs
    run_command(
        ["llamafactory-cli", "train", "configs/stage2_sft.yaml"],
        "Stage 2: Supervised Fine-Tuning on MedQA",
    )

    print("\nTraining complete!")
    print(f"Final adapter saved to: saves/llama3-8b-medqa/lora/sft")


if __name__ == "__main__":
    main()
