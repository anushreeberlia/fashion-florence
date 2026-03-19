"""Build train/val/test JSONL from labeling_progress.jsonl so you can train now and again later with more data.

Use when you have 10k+ rows in labeling_progress.jsonl and want to:
  1. Train Florence LoRA now on current data.
  2. Later, add more labels (resume generate_gpt_labels), then re-run this script and train again.

Usage:
  python training/build_splits_from_progress.py \\
    /path/to/labeling_progress.jsonl \\
    --out-dir /path/to/splits \\
    --seed 42

Output: out_dir/train.jsonl, out_dir/val.jsonl, out_dir/test.jsonl (same format train_lora expects).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_progress(progress_path: Path) -> list[dict]:
    """Load progress JSONL; dedupe by _item_id (last wins)."""
    if not progress_path.exists():
        print(f"ERROR: Not found: {progress_path}", file=sys.stderr)
        sys.exit(1)
    by_id: dict[str, dict] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        iid = row.get("_item_id", "")
        if iid:
            by_id[iid] = row
    return list(by_id.values())


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build train/val/test from labeling_progress.jsonl for LoRA training."
    )
    p.add_argument(
        "progress_file",
        type=Path,
        help="Path to labeling_progress.jsonl (e.g. .../fashion-florence-v2/labeling_progress.jsonl)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for train.jsonl, val.jsonl, test.jsonl. Default: same dir as progress file.",
    )
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = args.out_dir or args.progress_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_progress(args.progress_file)
    if not rows:
        print("ERROR: No valid rows in progress file.", file=sys.stderr)
        sys.exit(1)

    # Drop internal keys; train_lora expects image_path, prompt, target
    out_rows = []
    for r in rows:
        out_rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)

    n = len(out_rows)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = out_rows[:n_train]
    val = out_rows[n_train : n_train + n_val]
    test = out_rows[n_train + n_val :]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split)} rows → {path}")

    print(f"\nTotal: {n} rows. Train now with:")
    print(f"  python training/train_lora.py \\")
    print(f"    --train-file {out_dir / 'train.jsonl'} \\")
    print(f"    --val-file {out_dir / 'val.jsonl'} \\")
    print(f"    --output-dir <your-lora-output>")
    print("\nWhen you have more labeled data, re-run this script on the updated progress file, then train again.")


if __name__ == "__main__":
    main()
