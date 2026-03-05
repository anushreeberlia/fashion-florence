"""Build train/val/test JSONL manifests for Fashion Florence training.

Supports:
1) Local iMaterialist-style annotations + images
2) Hugging Face datasets (for example Marqo/iMaterialist)
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    # Works when invoked as a module: python -m training.prepare_dataset
    from training.label_mapping import (
        LABEL_ID_TO_NAME, convert_imat_labels, convert_hf_fields, format_as_training_target,
    )
except ModuleNotFoundError:
    # Works when invoked as a script path: python training/prepare_dataset.py
    from label_mapping import (
        LABEL_ID_TO_NAME, convert_imat_labels, convert_hf_fields, format_as_training_target,
    )

DEFAULT_PROMPT = "Analyze this clothing item image and return structured fashion tags as JSON."
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
GROUP_TO_SOURCE_KEYS = {
    "category": ("category",),
    "color": ("color",),
    "material": ("material",),
    "style": ("style",),
    "pattern": ("pattern",),
    "sleeve": ("sleeve",),
    "neckline": ("neckline",),
    "gender": ("gender",),
}


def _build_group_indices():
    by_name_group: dict[tuple[str, str], int] = {}
    by_group_names: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for label_id, (name, group) in LABEL_ID_TO_NAME.items():
        key = (name.lower(), group)
        by_name_group[key] = label_id
        by_group_names[group].append((name.lower(), label_id))

    for group in by_group_names:
        by_group_names[group].sort(key=lambda x: len(x[0]), reverse=True)
    return by_name_group, dict(by_group_names)


NAME_GROUP_TO_ID, GROUP_TO_NAMES = _build_group_indices()


def _pick_first(record: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _normalize_image_id(value) -> str:
    return str(value).strip()


def _normalize_label_ids(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = [value]
    normalized: list[int] = []
    for item in raw:
        try:
            normalized.append(int(item))
        except (TypeError, ValueError):
            continue
    return normalized


def _read_annotations(annotations_path: Path) -> tuple[dict[str, set[int]], dict[str, str]]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))

    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Expected top-level key 'annotations' as a list")

    image_to_labels: dict[str, set[int]] = defaultdict(set)
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = _pick_first(ann, ("image_id", "imageId", "id"))
        if image_id is None:
            continue
        label_ids = _normalize_label_ids(_pick_first(ann, ("label_id", "labelId", "labels")))
        image_id_str = _normalize_image_id(image_id)
        for lid in label_ids:
            image_to_labels[image_id_str].add(lid)

    image_to_filename: dict[str, str] = {}
    images = payload.get("images")
    if isinstance(images, list):
        for image_rec in images:
            if not isinstance(image_rec, dict):
                continue
            image_id = _pick_first(image_rec, ("image_id", "imageId", "id"))
            if image_id is None:
                continue
            file_name = _pick_first(
                image_rec,
                ("file_name", "fileName", "image_path", "imagePath", "path"),
            )
            if isinstance(file_name, str) and file_name.strip():
                image_to_filename[_normalize_image_id(image_id)] = file_name.strip()

    return image_to_labels, image_to_filename


def _build_basename_index(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def _resolve_image_path(
    image_id: str,
    image_to_filename: dict[str, str],
    images_dir: Path,
    basename_index: dict[str, Path] | None,
) -> tuple[Path | None, dict[str, Path] | None]:
    candidates: list[Path] = []

    file_name = image_to_filename.get(image_id)
    if file_name:
        file_path = Path(file_name)
        if file_path.is_absolute():
            candidates.append(file_path)
        else:
            candidates.append(images_dir / file_path)
            candidates.append(images_dir / file_path.name)

    candidates.append(images_dir / image_id)
    for ext in IMAGE_EXTENSIONS:
        candidates.append(images_dir / f"{image_id}{ext}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate, basename_index

    if file_name:
        if basename_index is None:
            basename_index = _build_basename_index(images_dir)
        basename_match = basename_index.get(Path(file_name).name)
        if basename_match and basename_match.exists() and basename_match.is_file():
            return basename_match, basename_index

    return None, basename_index


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _parse_value_to_label_ids(raw_value, group: str) -> list[int]:
    if raw_value is None:
        return []
    value = str(raw_value).strip()
    if not value:
        return []

    exact = NAME_GROUP_TO_ID.get((value.lower(), group))
    if exact is not None:
        return [exact]

    # Try common list separators first.
    split_parts = [p.strip() for p in re.split(r"[,/;|]+", value) if p.strip()]
    if split_parts and len(split_parts) > 1:
        ids: list[int] = []
        for part in split_parts:
            part_exact = NAME_GROUP_TO_ID.get((part.lower(), group))
            if part_exact is not None:
                ids.append(part_exact)
                continue
            ids.extend(_parse_value_to_label_ids(part, group))
        return sorted(set(ids))

    # Fallback: longest-name phrase matching for values like "Black White".
    lowered = f" {value.lower()} "
    matches: list[int] = []
    for name, label_id in GROUP_TO_NAMES.get(group, []):
        pattern = rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])"
        if re.search(pattern, lowered):
            matches.append(label_id)
    return sorted(set(matches))


def _label_ids_from_hf_row(row: dict) -> list[int]:
    label_ids: list[int] = []
    for group, keys in GROUP_TO_SOURCE_KEYS.items():
        for key in keys:
            if key not in row:
                continue
            label_ids.extend(_parse_value_to_label_ids(row.get(key), group))
    return sorted(set(label_ids))


def _save_hf_image(image_obj, image_path: Path) -> bool:
    if image_obj is None:
        return False
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = image_obj.convert("RGB")
        img.save(image_path, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def _iter_hf_rows(args: argparse.Namespace):
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install with: pip install datasets pillow"
        ) from exc

    dataset = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=50_000)
    if args.max_rows > 0:
        return itertools.islice(dataset, args.max_rows)
    return dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _split_rows(rows: list[dict], train_ratio: float, val_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val : n_train + n_val + n_test]
    return train_rows, val_rows, test_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Fashion Florence JSONL datasets.")
    parser.add_argument("--annotations", help="Path to iMaterialist annotations JSON.")
    parser.add_argument("--images-dir", help="Root directory containing images.")
    parser.add_argument("--hf-dataset", help="Hugging Face dataset id (e.g. Marqo/iMaterialist).")
    parser.add_argument("--hf-split", default="data", help="Hugging Face split name.")
    parser.add_argument("--max-rows", type=int, default=10000, help="Max rows to export from HF dataset.")
    parser.add_argument(
        "--hf-images-dir",
        default="data/raw/images",
        help="Where to save downloaded HF images (repo-relative or absolute).",
    )
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for JSONL and stats.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hf_dataset and not (args.annotations and args.images_dir):
        raise ValueError("Provide either --hf-dataset or both --annotations and --images-dir.")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    category_counts: Counter = Counter()
    skipped_rows = 0
    missing_images = 0
    invalid_target_rows = 0
    raw_rows = 0

    if args.hf_dataset:
        hf_images_dir = Path(args.hf_images_dir).expanduser()
        if not hf_images_dir.is_absolute():
            hf_images_dir = (repo_root / hf_images_dir).resolve()
        hf_images_dir.mkdir(parents=True, exist_ok=True)

        used_paths: set[str] = set()
        dataset = _iter_hf_rows(args)
        for idx, row in enumerate(dataset):
            raw_rows += 1
            mapped = convert_hf_fields(
                category=row.get("category"),
                color=row.get("color"),
                material=row.get("material"),
                style=row.get("style"),
            )
            if mapped is None:
                skipped_rows += 1
                continue

            item_id = row.get("item_ID") if isinstance(row, dict) else None
            image_stem = str(item_id) if item_id is not None else f"row_{idx}"
            image_path = hf_images_dir / f"{image_stem}.jpg"
            if image_path.as_posix() in used_paths:
                image_path = hf_images_dir / f"{image_stem}_{idx}.jpg"
            used_paths.add(image_path.as_posix())

            image_ok = _save_hf_image(row.get("image"), image_path)
            if not image_ok:
                missing_images += 1
                continue

            try:
                target = format_as_training_target(mapped)
            except ValueError:
                invalid_target_rows += 1
                continue
            records.append(
                {
                    "image_path": _to_repo_relative(image_path, repo_root),
                    "prompt": args.prompt,
                    "target": target,
                }
            )
            category_counts[mapped["category"]] += 1
    else:
        annotations_path = Path(args.annotations).expanduser().resolve()
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        images_dir = Path(args.images_dir).expanduser().resolve()
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        image_to_labels, image_to_filename = _read_annotations(annotations_path)
        raw_rows = len(image_to_labels)
        basename_index: dict[str, Path] | None = None

        for image_id, label_set in image_to_labels.items():
            mapped = convert_imat_labels(sorted(label_set))
            if mapped is None:
                skipped_rows += 1
                continue

            image_path, basename_index = _resolve_image_path(
                image_id=image_id,
                image_to_filename=image_to_filename,
                images_dir=images_dir,
                basename_index=basename_index,
            )
            if image_path is None:
                missing_images += 1
                continue

            try:
                target = format_as_training_target(mapped)
            except ValueError:
                invalid_target_rows += 1
                continue
            records.append(
                {
                    "image_path": _to_repo_relative(image_path, repo_root),
                    "prompt": args.prompt,
                    "target": target,
                }
            )
            category_counts[mapped["category"]] += 1

    rng = random.Random(args.seed)
    rng.shuffle(records)
    train_rows, val_rows, test_rows = _split_rows(records, args.train_ratio, args.val_ratio)

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    stats = {
        "source": args.hf_dataset or "local_annotations",
        "raw_rows": raw_rows,
        "kept_rows": len(records),
        "skipped_rows": skipped_rows,
        "missing_images": missing_images,
        "invalid_target_rows": invalid_target_rows,
        "split_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "category_counts": {
            key: category_counts.get(key, 0)
            for key in ("top", "bottom", "dress", "layer", "shoes", "accessory")
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Dataset prep complete:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
