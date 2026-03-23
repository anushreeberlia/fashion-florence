"""
Inference module for Fashion Florence (V1).

The model predicts 4 fields from a clothing image:
  category, primary_color, material, style_tags

This module adds 4 more fields via rule-based post-processing:
  secondary_colors (empty — V1 doesn't predict this)
  fit             (defaults to "regular")
  occasion_tags   (inferred from style_tags + category)
  season_tags     (inferred from material)

Usage (local):
    from training.inference import FashionFlorence
    ff = FashionFlorence.from_hub(device="cuda")
    tags = ff.predict("path/to/image.jpg")

Usage (HuggingFace Inference API):
    from training.inference import predict_with_postprocess
    raw = call_hf_api(image)  # your HF API call
    tags = predict_with_postprocess(raw)
"""

from __future__ import annotations

import json
import re
from typing import Any

MODEL_ID = "anushreeberlia/fashion-florence"

# ── Rule tables ──────────────────────────────────────────────────────────────

_STYLE_TO_OCCASION: dict[str, list[str]] = {
    "sporty":     ["gym", "workout"],
    "athletic":   ["gym", "workout"],
    "activewear": ["gym", "workout"],
    "sexy":       ["party", "night-out"],
    "glamorous":  ["party", "night-out"],
    "edgy":       ["going-out", "night-out"],
    "elegant":    ["dinner", "date", "formal-event"],
    "classic":    ["work", "everyday"],
    "workwear":   ["work"],
    "preppy":     ["work", "everyday"],
    "bohemian":   ["vacation", "casual"],
    "streetwear": ["everyday", "casual"],
    "minimalist": ["everyday", "work"],
    "casual":     ["everyday", "casual"],
    "trendy":     ["everyday", "going-out"],
    "chic":       ["everyday", "date"],
    "vintage":    ["everyday", "casual"],
    "statement":  ["party", "going-out"],
}

_CATEGORY_OCCASION_DEFAULTS: dict[str, list[str]] = {
    "shoes":     ["everyday"],
    "accessory": ["everyday"],
    "layer":     ["everyday", "casual"],
}

_WARM_MATERIALS = {"wool", "cashmere", "fleece", "tweed", "velvet", "velour", "flannel", "corduroy", "knit"}
_COOL_MATERIALS = {"linen", "chiffon", "organza", "tulle"}
_LIGHT_MATERIALS = {"lace", "satin", "silk"}

_CATEGORY_FIT: dict[str, str] = {
    "top":       "regular",
    "bottom":    "regular",
    "dress":     "regular",
    "layer":     "relaxed",
    "shoes":     "regular",
    "accessory": "regular",
}


def infer_occasion_tags(category: str, style_tags: list[str]) -> list[str]:
    """Derive occasion_tags from category and style_tags."""
    occasions: set[str] = set()

    for style in style_tags:
        for tag in _STYLE_TO_OCCASION.get(style, []):
            occasions.add(tag)

    if not occasions:
        for tag in _CATEGORY_OCCASION_DEFAULTS.get(category, ["everyday", "casual"]):
            occasions.add(tag)

    return sorted(occasions)


def infer_season_tags(material: str) -> list[str]:
    """Derive season_tags from material."""
    mat = material.lower().strip() if material else ""

    if mat in _WARM_MATERIALS:
        return ["fall", "winter"]
    if mat in _COOL_MATERIALS:
        return ["spring", "summer"]
    if mat in _LIGHT_MATERIALS:
        return ["spring", "summer", "fall"]
    return ["all_season"]


def infer_fit(category: str) -> str:
    """Default fit based on category."""
    return _CATEGORY_FIT.get(category, "regular")


# ── JSON parsing ─────────────────────────────────────────────────────────────

_V1_FIELDS = ["category", "primary_color", "material", "style_tags"]


def _parse_model_output(raw: str) -> dict[str, Any] | None:
    """Parse V1 model output (4-field JSON) into a dict."""
    text = raw.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for suffix in ["}", "]}", "\"]}"]:
        try:
            obj = json.loads(text + suffix)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # Regex extraction as last resort
    result: dict[str, Any] = {}
    for field in _V1_FIELDS:
        str_match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
        if str_match:
            result[field] = str_match.group(1)
            continue
        arr_match = re.search(rf'"{field}"\s*:\s*\[([^\]]*)\]', text)
        if arr_match:
            inner = arr_match.group(1)
            items = [s.strip().strip('"').strip("'") for s in inner.split(",") if s.strip().strip('"')]
            result[field] = items

    return result if result else None


# ── Post-processing pipeline ─────────────────────────────────────────────────

def predict_with_postprocess(raw_model_output: str) -> dict[str, Any]:
    """
    Parse V1 model output (4 fields) and expand to 8 fields.

    Model provides:  category, primary_color, material, style_tags
    Rules add:       secondary_colors, fit, occasion_tags, season_tags
    """
    parsed = _parse_model_output(raw_model_output) or {}

    category = parsed.get("category", "unknown")
    primary_color = parsed.get("primary_color", "unknown")
    material = parsed.get("material", "unknown")
    style_tags = parsed.get("style_tags", ["casual"])

    if isinstance(style_tags, str):
        style_tags = [style_tags]
    if not style_tags:
        style_tags = ["casual"]

    return {
        "category": category,
        "primary_color": primary_color,
        "secondary_colors": [],
        "material": material,
        "fit": infer_fit(category),
        "style_tags": style_tags,
        "occasion_tags": infer_occasion_tags(category, style_tags),
        "season_tags": infer_season_tags(material),
    }


# ── Full local inference class ───────────────────────────────────────────────

class FashionFlorence:
    """Load Fashion Florence V1 and run inference with post-processing."""

    PROMPT = "Analyze this clothing item image and return structured fashion tags as JSON."

    def __init__(self, processor, model, device: str = "cuda"):
        self.processor = processor
        self.model = model
        self.device = device

    @classmethod
    def from_hub(
        cls,
        model_id: str = MODEL_ID,
        device: str = "cuda",
    ) -> "FashionFlorence":
        """Load the merged V1 model directly from HuggingFace."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, trust_remote_code=True,
        ).eval().to(device)
        return cls(processor, model, device)

    def predict_raw(self, image_path: str) -> str:
        """Run model inference, return raw decoded string."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=self.PROMPT, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]

    def predict(self, image_path: str) -> dict[str, Any]:
        """Run inference and return post-processed 8-field tags."""
        return predict_with_postprocess(self.predict_raw(image_path))


if __name__ == "__main__":
    print("=== V1 output (4 fields) → expanded to 8 fields ===\n")

    v1_output = '{"category":"dress","primary_color":"red","material":"silk","style_tags":["elegant","glamorous"]}'
    result = predict_with_postprocess(v1_output)
    print(f"V1 raw: {v1_output}\n")
    print(f"Expanded:\n{json.dumps(result, indent=2)}")
