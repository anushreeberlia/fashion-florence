"""Standalone schema definition for Fashion Florence training targets.

Open-ended schema: category is the only strictly constrained field (6 buckets).
Colors, materials, styles, and occasions are free-form — the model learns
whatever values appear in the training data and can generalize at inference.
"""

from __future__ import annotations


ALLOWED_CATEGORIES = {
    "top",
    "bottom",
    "dress",
    "layer",
    "shoes",
    "accessory",
}


def validate_training_target(labels: dict) -> None:
    """
    Validate structural requirements only.
    Category must be one of 6 buckets; all other fields are free-form strings/lists.
    """
    required_keys = {"category", "primary_color", "material", "style_tags"}
    missing = sorted(required_keys - set(labels.keys()))
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    if labels["category"] not in ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid category: {labels['category']}")
    if not isinstance(labels["primary_color"], str) or not labels["primary_color"].strip():
        raise ValueError("primary_color must be a non-empty string")
    if not isinstance(labels["material"], str) or not labels["material"].strip():
        raise ValueError("material must be a non-empty string")
    if not isinstance(labels["style_tags"], list) or not labels["style_tags"]:
        raise ValueError("style_tags must be a non-empty list")
