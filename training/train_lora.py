"""LoRA fine-tuning script for Florence-2 on prepared JSONL data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Florence-2 LoRA adapters.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="microsoft/Florence-2-base")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=256)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class FlorenceJsonlDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict],
        processor,
        max_input_length: int,
        max_target_length: int,
        repo_root: Path,
    ):
        self.rows = rows
        self.processor = processor
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.repo_root = repo_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.repo_root / image_path

        image = Image.open(image_path).convert("RGB")
        prompt = row["prompt"]
        target = row["target"]

        model_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )

        target_ids = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        )["input_ids"].squeeze(0)

        target_ids[target_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "pixel_values": model_inputs["pixel_values"].squeeze(0),
            "labels": target_ids,
        }


@dataclass
class DataCollator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    # Some environments resolve Florence tokenizer as RobertaTokenizer.
    tok = processor.tokenizer
    if not hasattr(tok, "image_token"):
        if "<image>" not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
        tok.image_token = "<image>"
        tok.image_token_id = tok.convert_tokens_to_ids("<image>")
        model.resize_token_embeddings(len(tok))

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    repo_root = Path.cwd()

    train_dataset = FlorenceJsonlDataset(
        rows=train_rows,
        processor=processor,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        repo_root=repo_root,
    )
    val_dataset = FlorenceJsonlDataset(
        rows=val_rows,
        processor=processor,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        repo_root=repo_root,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=False,
        weight_decay=0.01,
        warmup_ratio=0.05,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA artifacts + processor to: {args.output_dir}")


if __name__ == "__main__":
    main()
