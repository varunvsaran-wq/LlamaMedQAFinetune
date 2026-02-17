# Fine-tuning Llama 3 8B on MedQA (USMLE)

Fine-tune Meta Llama 3 8B for medical question answering using the MedQA dataset with a two-stage QLoRA approach, optimized for 8GB VRAM (RTX 4050).

## Overview

**Stage 1 — Continued Pretraining**: Adapt the base model to the medical domain using 18 English medical textbooks (e.g., Harrison's, Robbins, First Aid).

**Stage 2 — Supervised Fine-Tuning**: Train on ~10K USMLE-style multiple-choice QA pairs to learn the task format and clinical reasoning.

Both stages use QLoRA (4-bit NF4 quantization) with LoRA rank 16 to fit within 8GB VRAM.

## Dataset

- **MedQA (USMLE)**: 10,178 training / dev / test QA pairs in JSONL format
- **Medical Textbooks**: 18 English textbooks covering anatomy, pathology, pharmacology, internal medicine, surgery, and more

## Project Structure

```
├── requirements.txt                # Dependencies
├── data_clean/                     # Raw MedQA data
├── data/                           # LLaMA-Factory formatted data (generated)
│   ├── dataset_info.json           # Dataset registry
│   ├── medqa_sft.jsonl             # Formatted QA pairs
│   └── textbooks_pretrain.jsonl    # Chunked textbook text
├── scripts/
│   ├── preprocess_qa.py            # Convert QA -> Alpaca format
│   └── preprocess_textbooks.py     # Chunk textbooks -> JSONL
├── configs/
│   ├── stage1_pretrain.yaml        # Continued pretraining config
│   └── stage2_sft.yaml             # SFT config
├── train.py                        # Orchestrator (runs both stages)
└── evaluate.py                     # Perplexity + accuracy evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

You need access to `meta-llama/Meta-Llama-3-8B` on Hugging Face. Log in with:

```bash
huggingface-cli login
```

## Usage

### Preprocess Data

```bash
python scripts/preprocess_qa.py
python scripts/preprocess_textbooks.py
```

### Train (both stages)

```bash
python train.py
```

Or run stages individually:

```bash
llamafactory-cli train configs/stage1_pretrain.yaml
llamafactory-cli train configs/stage2_sft.yaml
```

### Evaluate

```bash
python evaluate.py
```

## Design Choices

| Choice | Rationale |
|--------|-----------|
| QLoRA 4-bit NF4 + double quant | Fits 8B model in 8GB VRAM |
| LoRA rank 16, alpha 32 | Good capacity without OOM |
| Textbook pretraining first | Builds medical knowledge before task training |
| 1024 token cutoff | Balances context vs. VRAM |
| paged_adamw_8bit | Prevents OOM spikes during optimizer step |
| Two-stage LoRA | Stage 1 adapter loaded into Stage 2 for cumulative learning |
