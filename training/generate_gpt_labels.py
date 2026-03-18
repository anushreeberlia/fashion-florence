"""Generate 8-field training labels using GPT-4o mini as a teacher model.

Downloads a fresh slice of iMaterialist (non-overlapping with V1 training data),
sends each image to GPT-4o mini with Loom's full tagging prompt, and saves
the results as JSONL for Florence V2 training.

V1 used seed=42, max_rows=50000. This script uses --offset to skip past those
rows in the shuffled stream, guaranteeing zero overlap.

Usage:
    # Colab: put key in a file (e.g. Drive), then:
    python training/generate_gpt_labels.py --api-key-file /content/drive/MyDrive/openai_key.txt \
        --out-dir ...
    # Or: export OPENAI_API_KEY=sk-... && python training/generate_gpt_labels.py \
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

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

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
OPENAI_VISION_MODEL = "gpt-4o-mini"


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
    p.add_argument("--batch-save-every", type=int, default=50,
                   help="Append new rows to progress file every N items (never truncates old rows).")
    p.add_argument("--resume", action="store_true",
                   help="Deprecated: resume is automatic when labeling_progress.jsonl exists. Ignored.")
    p.add_argument("--fresh", action="store_true",
                   help="Delete existing labeling_progress.jsonl and start from zero (use with care).")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--cost-limit", type=float, default=15.0,
                   help="Stop if estimated cost exceeds this (USD).")
    p.add_argument(
        "--api-key-file",
        default=None,
        metavar="PATH",
        help="Text file with OpenAI API key on first line (safer than --api-key on Colab).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="OpenAI API key inline (shows in shell history — use --api-key-file if possible).",
    )
    p.add_argument(
        "--use-responses",
        action="store_true",
        help="Use /v1/responses (can error on some accounts). Default: chat/completions (stable for vision).",
    )
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


def _responses_output_text(data: dict) -> str | None:
    """Extract assistant text from POST /v1/responses JSON."""
    err = data.get("error")
    if err:
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.warning(f"OpenAI response error: {msg}")
        return None
    for block in data.get("output") or []:
        if block.get("type") != "message":
            continue
        for part in block.get("content") or []:
            if part.get("type") == "output_text":
                return (part.get("text") or "").strip()
    return None


def _sanitize_api_key(raw: str) -> str:
    k = (raw or "").strip()
    if k.startswith("\ufeff"):
        k = k[1:].strip()
    if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
        k = k[1:-1].strip()
    return k


def _gpt_parse_json_text(text: str) -> dict | None:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1])
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def _call_openai_chat(client, image_b64: str) -> str | None:
    """Chat Completions + vision (stable). Returns assistant text or None."""
    r = client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[
            {"role": "system", "content": GPT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GPT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )
    msg = r.choices[0].message.content
    return (msg or "").strip() if msg else None


def _call_openai_responses(client, image_b64: str) -> str | None:
    """Responses API. Returns assistant text or None."""
    payload_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": GPT_PROMPT},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "low",
                },
            ],
        }
    ]
    r = client.responses.create(
        model=OPENAI_VISION_MODEL,
        instructions=GPT_SYSTEM,
        input=payload_input,
        temperature=0.2,
        max_output_tokens=500,
    )
    data = r.model_dump() if hasattr(r, "model_dump") else {}
    text = _responses_output_text(data)
    if not text and hasattr(r, "output_text"):
        text = (getattr(r, "output_text", None) or "").strip()
    return text or None


def call_gpt_vision(
    image_b64: str,
    api_key: str,
    retries: int = 3,
    *,
    use_responses: bool = False,
) -> dict | None:
    """Vision + JSON tags. Default: chat/completions. Pass use_responses=True for /v1/responses."""
    key = _sanitize_api_key(api_key)
    if not key:
        logger.error("OpenAI API key is empty — use --api-key, --api-key-file, or OPENAI_API_KEY")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Run: pip install -q openai")
        return None

    client = OpenAI(api_key=key, base_url="https://api.openai.com/v1")
    backend = "responses" if use_responses else "chat"

    for attempt in range(retries):
        try:
            if use_responses:
                text = _call_openai_responses(client, image_b64)
            else:
                text = _call_openai_chat(client, image_b64)
            if not text:
                logger.warning(f"No text from OpenAI ({backend})")
                time.sleep(2)
                continue
            parsed = _gpt_parse_json_text(text)
            if parsed is None:
                logger.warning(f"Attempt {attempt + 1}: model returned non-JSON ({backend})")
                time.sleep(2)
                continue
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt + 1} JSON parse failed: {e}")
            time.sleep(2)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in str(e):
                wait = min(2 ** (attempt + 1), 30)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if "401" in str(e) or "bearer" in err or "authentication" in err or "api key" in err:
                logger.error(
                    "OpenAI auth failed. Check key at platform.openai.com/api-keys; "
                    "use --api-key-file with one line sk-... (no quotes)."
                )
            logger.warning(f"Attempt {attempt + 1} failed ({backend}): {e}")
            time.sleep(2)

    return None


def _fsync_path(path: Path) -> None:
    """Force-flush a file to disk (critical for Colab's Drive FUSE mount)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


def _load_progress(progress_file: Path) -> tuple[list[dict], set[str]]:
    """Load progress; dedupe by _item_id (last line wins). Skips bad JSON lines."""
    if not progress_file.exists():
        return [], set()
    by_id: dict[str, dict] = {}
    bad = 0
    for i, line in enumerate(progress_file.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if not isinstance(row, dict):
            continue
        iid = row.get("_item_id", "")
        if iid:
            by_id[iid] = row
    if bad:
        logger.warning(f"Skipped {bad} bad line(s) in {progress_file.name}")
    records = list(by_id.values())
    return records, set(by_id.keys())


def _append_progress(progress_file: Path, records: list[dict], start: int) -> int:
    """Append records[start:] to JSONL (never truncates). Returns new start index."""
    if start >= len(records):
        return start
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "a", encoding="utf-8") as f:
        for r in records[start:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    _fsync_path(progress_file)
    _fsync_path(progress_file.parent)
    return len(records)


def save_image(pil_image, path: Path) -> bool:
    """Save a PIL image or image bytes to disk."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(pil_image, Image.Image):
            pil_image.save(path, format="JPEG", quality=90)
        elif isinstance(pil_image, bytes):
            with open(path, "wb") as f:
                f.write(pil_image)
                f.flush()
                os.fsync(f.fileno())
        else:
            pil_image = pil_image.convert("RGB")
            pil_image.save(path, format="JPEG", quality=90)
        _fsync_path(path)
        return True
    except Exception as e:
        logger.warning(f"Failed to save image {path}: {e}")
        return False


def main() -> None:
    args = parse_args()

    api_key = ""
    if args.api_key:
        api_key = _sanitize_api_key(args.api_key)
    elif args.api_key_file:
        kpath = Path(args.api_key_file).expanduser()
        if not kpath.is_absolute():
            kpath = kpath.resolve()
        if not kpath.is_file():
            print(f"ERROR: --api-key-file not found: {kpath}", file=sys.stderr)
            sys.exit(1)
        lines = [ln.strip() for ln in kpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        api_key = _sanitize_api_key(lines[0]) if lines else ""
    else:
        api_key = _sanitize_api_key(os.getenv("OPENAI_API_KEY") or "")
    if not api_key:
        print(
            "ERROR: Pass --api-key, --api-key-file, or set OPENAI_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(api_key) < 20:
        print("ERROR: API key looks too short or wrong.", file=sys.stderr)
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info(f"OpenAI key OK (length {len(api_key)} chars)")
    if args.use_responses:
        logger.info("OpenAI backend: Responses API (/v1/responses)")
    else:
        logger.info("OpenAI backend: chat/completions (default; omit --use-responses if Responses errors)")

    # Empty HF_TOKEN makes the Hub send "Bearer " with no token → same JSON error as OpenAI 401
    for _hf_env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if not (os.environ.get(_hf_env) or "").strip():
            os.environ.pop(_hf_env, None)

    try:
        from openai import OpenAI

        _c = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        next(iter(_c.models.list()))
        logger.info("OpenAI auth OK (verified with models.list)")
    except Exception as e:
        es = str(e).lower()
        if "401" in str(e) or "bearer" in es or "authentication" in es or "invalid_api_key" in es:
            logger.error(
                "OpenAI rejected this API key. New key: https://platform.openai.com/api-keys — %s",
                e,
            )
            sys.exit(1)
        logger.warning("OpenAI models.list check inconclusive (%s); continuing.", e)

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

    # Verify Drive is actually writable (catches unmounted/fake FUSE paths)
    _probe = out_dir / ".write_probe"
    try:
        with open(_probe, "w") as f:
            f.write("ok")
            f.flush()
            os.fsync(f.fileno())
        _probe_content = _probe.read_text()
        _probe.unlink()
        if _probe_content != "ok":
            raise IOError("read-back mismatch")
        logger.info(f"Drive write verified: {out_dir}")
    except Exception as e:
        logger.error(f"FATAL: Cannot write to {out_dir}: {e}")
        logger.error("Is Google Drive mounted? Run: from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)

    progress_file = out_dir / "labeling_progress.jsonl"
    if args.fresh and progress_file.exists():
        bak = progress_file.with_suffix(".jsonl.bak")
        progress_file.rename(bak)
        logger.warning(f"--fresh: moved existing progress to {bak.name}")
    records, completed_ids = _load_progress(progress_file)
    append_from = len(records)
    if records:
        logger.info(
            f"RESUME: {len(records)} rows on disk — new labels will APPEND (old rows are never wiped). "
            f"Use --fresh to restart from zero."
        )
    else:
        logger.info("Starting new labeling_progress.jsonl (append-only checkpoints)")

    logger.info(f"Loading dataset: {args.hf_dataset} (offset={args.offset}, max={args.max_rows})")
    dataset = load_dataset(
        args.hf_dataset,
        split=args.hf_split,
        streaming=True,
        token=False,
    )
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
        tags = call_gpt_vision(
            image_b64,
            api_key,
            retries=args.max_retries,
            use_responses=args.use_responses,
        )
        if not isinstance(tags, dict):
            errors += 1
            logger.warning(f"  [{labeled}/{args.max_rows}] {item_id}: GPT returned non-dict, skipping")
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

        # Append every N *new* rows (not total labeled — works correctly on resume)
        pending = len(records) - append_from
        if pending >= args.batch_save_every:
            append_from = _append_progress(progress_file, records, append_from)
            logger.info(f"  Progress appended {pending} rows (total {len(records)} in memory / on disk)")

        if est_cost > args.cost_limit:
            logger.warning(f"Cost limit ${args.cost_limit} reached. Stopping.")
            break

    # Flush any remaining rows not yet appended
    if len(records) > append_from:
        _append_progress(progress_file, records, append_from)
        logger.info(f"  Final append: {len(records) - append_from} rows")

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
            f.flush()
            os.fsync(f.fileno())
        _fsync_path(path)
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
        f.flush()
        os.fsync(f.fileno())
    _fsync_path(stats_path)
    logger.info(f"\nDone! Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
