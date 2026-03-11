"""Generate 8-field training labels using GPT-4o mini as a teacher model.

Downloads a fresh slice of iMaterialist (non-overlapping with V1 training data),
sends each image to GPT-4o mini with Loom's full tagging prompt, and saves
the results as JSONL for Florence V2 training.

V1 used seed=42, max_rows=50000. This script uses --offset to skip past those
rows in the shuffled stream, guaranteeing zero overlap.

Usage:
    export OPENAI_API_KEY=sk-...
    python training/generate_gpt_labels.py \
        --max-rows 50000 \
        --offset 50000 \
        --out-dir data/processed_v2
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GPT_SYSTEM = "You are a fashion expert. Analyze clothing images and return structured JSON only."

GPT_PROMPT = """Analyze the clothing item in this image. Return ONLY valid JSON with these keys:

- category: one of [top, bottom, dress, layer, shoes, accessory]
  * dress = ANY one-piece garment (mini dress, midi dress, bodycon dress, maxi, jumpsuit, romper)
  * top = SEPARATE upper body pieces only (blouses, t-shirts, sweaters, tanks, crop tops)
  * layer = outerwear worn OVER other clothes (jackets, coats, blazers, cardigans)
  * bottom = SEPARATE lower body pieces (pants, jeans, skirts, shorts)
  * shoes = footwear
  * accessory = bags, jewelry, scarves, hats, belts
- primary_color: one of [black, white, gray, beige, brown, blue, navy, green, yellow, orange, red, pink, purple, metallic, multi, unknown]
- secondary_colors: array from same palette (can be empty)
- material: REQUIRED string - the fabric type (cotton, silk, knit, jersey, velvet, satin, leather, denim, linen, polyester, wool, chiffon, lace, etc.)
- fit: one of [fitted, bodycon, slim, straight, relaxed, oversized, wide, cropped, loose, unknown]
- style_tags: array from [minimalist, classic, edgy, romantic, sporty, athletic, activewear, bohemian, streetwear, preppy, elegant, casual, chic, vintage, statement, workwear, sexy, glamorous, trendy]
- occasion_tags: array from [everyday, casual, work, dinner, party, formal, vacation, lounge, wedding_guest, going-out, clubbing, gym, workout, date, night-out, brunch]
- season_tags: array - use EITHER ["all_season"] OR subset of [spring, summer, fall, winter], NEVER both

JSON only, no markdown."""

FLORENCE_PROMPT = "Analyze this clothing item image and return structured fashion tags as JSON."

MAX_IMAGE_SIZE = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GPT-4o mini labels for Florence V2 training.")
    p.add_argument("--max-rows", type=int, default=50000, help="Number of images to label.")
    p.add_argument("--offset", type=int, default=50000,
                   help="Skip this many rows in the shuffled stream (to avoid V1 overlap). "
                        "V1 used seed=42, max_rows=50000, so offset=50000 skips those.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (same as V1 to ensure deterministic skip).")
    p.add_argument("--hf-dataset", default="Marqo/iMaterialist")
    p.add_argument("--hf-split", default="data")
    p.add_argument("--images-dir", default="data/raw_v2/images")
    p.add_argument("--out-dir", default="data/processed_v2")
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--batch-save-every", type=int, default=100,
                   help="Write progress to disk every N items.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing progress file.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--cost-limit", type=float, default=15.0,
                   help="Stop if estimated cost exceeds this (USD).")
    return p.parse_args()


def resize_for_gpt(image_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    if max(img.size) > MAX_IMAGE_SIZE:
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def call_gpt_vision(image_b64: str, retries: int = 3) -> dict | None:
    """Send image to GPT-4o mini, return parsed JSON dict or None on failure."""
    for attempt in range(retries):
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": GPT_SYSTEM},
                        {"role": "user", "content": [
                            {"type": "text", "text": GPT_PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            }},
                        ]},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500,
                },
                timeout=30.0,
            )

            if resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 30)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                logger.warning(f"GPT error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
                continue

            text = resp.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)

        except (json.JSONDecodeError, KeyError, httpx.TimeoutException) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    return None


def save_image(pil_image, path: Path) -> bool:
    """Save a PIL image or image bytes to disk."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(pil_image, Image.Image):
            pil_image.save(path, format="JPEG", quality=90)
        elif isinstance(pil_image, bytes):
            with open(path, "wb") as f:
                f.write(pil_image)
        else:
            pil_image = pil_image.convert("RGB")
            pil_image.save(path, format="JPEG", quality=90)
        return True
    except Exception as e:
        logger.warning(f"Failed to save image {path}: {e}")
        return False


def main() -> None:
    args = parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    from datasets import load_dataset

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir)
    if not images_dir.is_absolute():
        images_dir = (repo_root / images_dir).resolve()
    images_dir.mkdir(parents=True, exist_ok=True)

    progress_file = out_dir / "labeling_progress.jsonl"
    completed_ids: set[str] = set()
    records: list[dict] = []

    if args.resume and progress_file.exists():
        for line in progress_file.read_text().strip().splitlines():
            row = json.loads(line)
            records.append(row)
            completed_ids.add(row.get("_item_id", ""))
        logger.info(f"Resumed: {len(records)} already labeled")

    logger.info(f"Loading dataset: {args.hf_dataset} (offset={args.offset}, max={args.max_rows})")
    dataset = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=50_000)

    # Skip past V1 rows
    stream = iter(dataset)
    logger.info(f"Skipping first {args.offset} rows (V1 training data)...")
    for _ in range(args.offset):
        try:
            next(stream)
        except StopIteration:
            print("ERROR: Dataset has fewer rows than offset.", file=sys.stderr)
            sys.exit(1)

    labeled = len(records)
    errors = 0
    est_cost = labeled * 0.0003  # ~$0.0003 per image at gpt-4o-mini with detail=low

    logger.info(f"Starting labeling from row {args.offset} → target {args.max_rows} rows")
    logger.info(f"Estimated cost: ~${args.max_rows * 0.0003:.2f}")

    for idx, row in enumerate(stream):
        if labeled >= args.max_rows:
            break

        item_id = str(row.get("item_ID", f"row_{args.offset + idx}"))
        if item_id in completed_ids:
            continue

        # Save image
        image_path = images_dir / f"{item_id}.jpg"
        if not image_path.exists():
            pil_img = row.get("image")
            if pil_img is None:
                errors += 1
                continue
            if not save_image(pil_img, image_path):
                errors += 1
                continue

        # Read and resize for GPT
        image_bytes = image_path.read_bytes()
        resized = resize_for_gpt(image_bytes)
        image_b64 = base64.b64encode(resized).decode("utf-8")

        # Call GPT-4o mini
        tags = call_gpt_vision(image_b64, retries=args.max_retries)
        if tags is None:
            errors += 1
            logger.warning(f"  [{labeled}/{args.max_rows}] {item_id}: GPT failed, skipping")
            continue

        required = {"category", "primary_color", "material", "style_tags"}
        if not required.issubset(tags.keys()):
            errors += 1
            logger.warning(f"  [{labeled}/{args.max_rows}] {item_id}: Missing fields, skipping")
            continue

        try:
            rel_path = str(image_path.relative_to(repo_root))
        except ValueError:
            rel_path = str(image_path)

        record = {
            "image_path": rel_path,
            "prompt": FLORENCE_PROMPT,
            "target": json.dumps(tags, ensure_ascii=False, separators=(",", ":")),
            "_item_id": item_id,
        }
        records.append(record)
        completed_ids.add(item_id)
        labeled += 1
        est_cost += 0.0003

        if labeled % 10 == 0:
            logger.info(f"  [{labeled}/{args.max_rows}] {item_id}: {tags.get('category')} "
                        f"| cost ~${est_cost:.2f}")

        # Periodic save
        if labeled % args.batch_save_every == 0:
            with open(progress_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"  Progress saved ({labeled} rows)")

        if est_cost > args.cost_limit:
            logger.warning(f"Cost limit ${args.cost_limit} reached. Stopping.")
            break

    # Final save of progress
    with open(progress_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Split into train/val/test
    import random
    rng = random.Random(args.seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    for split_name, split_rows in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for r in split_rows:
                row_out = {k: v for k, v in r.items() if not k.startswith("_")}
                f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
        logger.info(f"  {split_name}: {len(split_rows)} rows → {path}")

    # Stats
    stats = {
        "total_labeled": labeled,
        "errors": errors,
        "estimated_cost_usd": round(est_cost, 2),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "offset_used": args.offset,
        "seed": args.seed,
        "v1_overlap": "none (skipped first 50k rows in seed=42 shuffle)",
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nDone! Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
