"""Standalone schema definition for Fashion Florence training targets."""

from __future__ import annotations


ALLOWED_CATEGORIES = {
    "top",
    "bottom",
    "dress",
    "layer",
    "shoes",
    "accessory",
}

ALLOWED_PRIMARY_COLORS = {
    "black",
    "white",
    "gray",
    "beige",
    "brown",
    "blue",
    "navy",
    "green",
    "yellow",
    "orange",
    "red",
    "pink",
    "purple",
    "metallic",
    "multi",
    "unknown",
}

ALLOWED_STYLE_TAGS = {
    "minimalist",
    "classic",
    "edgy",
    "romantic",
    "sporty",
    "athletic",
    "activewear",
    "bohemian",
    "streetwear",
    "preppy",
    "elegant",
    "casual",
    "chic",
    "vintage",
    "statement",
    "workwear",
    "sexy",
    "glamorous",
    "trendy",
}

ALLOWED_OCCASION_TAGS = {
    "everyday",
    "casual",
    "work",
    "dinner",
    "party",
    "formal",
    "vacation",
    "lounge",
    "wedding_guest",
    "going-out",
    "clubbing",
    "gym",
    "workout",
    "date",
    "night-out",
    "brunch",
}


def _assert_known_tags(tags: list[str], allowed: set[str], field: str) -> None:
    unknown = sorted(set(tags) - allowed)
    if unknown:
        raise ValueError(f"Unknown {field}: {unknown}")


def validate_training_target(labels: dict) -> None:
    """
    Validate mapped labels against Fashion Florence's standalone schema.
    Raises ValueError when the payload is invalid.
    """
    required_keys = {
        "category",
        "primary_color",
        "material",
        "style_tags",
        "occasion_tags",
    }
    missing = sorted(required_keys - set(labels.keys()))
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    category = labels["category"]
    primary_color = labels["primary_color"]
    material = labels["material"]
    style_tags = labels["style_tags"]
    occasion_tags = labels["occasion_tags"]

    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid category: {category}")
    if primary_color not in ALLOWED_PRIMARY_COLORS:
        raise ValueError(f"Invalid primary_color: {primary_color}")
    if not isinstance(material, str) or not material.strip():
        raise ValueError("material must be a non-empty string")
    if not isinstance(style_tags, list) or not style_tags:
        raise ValueError("style_tags must be a non-empty list")
    if not isinstance(occasion_tags, list) or not occasion_tags:
        raise ValueError("occasion_tags must be a non-empty list")

    _assert_known_tags(style_tags, ALLOWED_STYLE_TAGS, "style_tags")
    _assert_known_tags(occasion_tags, ALLOWED_OCCASION_TAGS, "occasion_tags")
