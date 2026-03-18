"""Normalize image_path in labeling_progress.jsonl so all point to .../images/ (not .../images/images/).

Use when you have files in both fashion-florence-v2/images and fashion-florence-v2/images/images.
Run this, then move contents of images/images/ into images/ (overwriting duplicates is fine).

Usage:
  python training/normalize_image_paths.py /path/to/fashion-florence-v2/labeling_progress.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python training/normalize_image_paths.py <labeling_progress.jsonl>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    changed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        img = row.get("image_path", "")
        if isinstance(img, str) and "/images/images/" in img:
            row["image_path"] = img.replace("/images/images/", "/images/")
            changed += 1
        out.append(json.dumps(row, ensure_ascii=False))

    if changed == 0:
        print("No paths contained '/images/images/'; nothing changed.")
        return

    backup = path.with_suffix(".jsonl.bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Updated {changed} paths; backup at {backup.name}")


if __name__ == "__main__":
    main()
