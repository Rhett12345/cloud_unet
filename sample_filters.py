"""Shared patch/sample supervision filtering helpers."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

import config as cfg


def _resolve_rule(mode: str) -> Dict[str, float]:
    mode = (mode or "train").lower()
    rules = getattr(cfg, "PATCH_FILTER_RULES", {}) or {}
    base = dict(rules.get("default", {}))
    base.update(rules.get(mode, {}))
    return base


def _threshold(min_pixels: float, min_ratio: float, patch_area: int) -> int:
    min_pixels = int(min_pixels or 0)
    min_ratio = float(min_ratio or 0.0)
    return max(min_pixels, int(math.ceil(patch_area * min_ratio)))


def get_patch_supervision_thresholds(mode: str, patch_size: Tuple[int, int]) -> Dict[str, int]:
    ph, pw = patch_size
    area = ph * pw
    rule = _resolve_rule(mode)
    return {
        "min_valid_label_pixels": _threshold(
            rule.get("min_valid_label_pixels", 0),
            rule.get("min_valid_label_ratio", 0.0),
            area,
        ),
        "min_valid_cloudy_pixels": _threshold(
            rule.get("min_valid_cloudy_pixels", 0),
            rule.get("min_valid_cloudy_ratio", 0.0),
            area,
        ),
    }


def count_supervision_pixels(
    patch_clp: np.ndarray,
    patch_cer: np.ndarray,
    patch_cot: np.ndarray,
    patch_cth: np.ndarray,
) -> Dict[str, int]:
    valid_label = np.isfinite(patch_clp)
    valid_cloudy = (
        valid_label
        & (patch_clp > 0)
        & np.isfinite(patch_cer)
        & np.isfinite(patch_cot)
        & np.isfinite(patch_cth)
    )
    return {
        "valid_label_pixels": int(valid_label.sum()),
        "valid_cloudy_pixels": int(valid_cloudy.sum()),
    }


def patch_passes_supervision(
    patch_clp: np.ndarray,
    patch_cer: np.ndarray,
    patch_cot: np.ndarray,
    patch_cth: np.ndarray,
    mode: str,
    patch_size: Tuple[int, int],
) -> Tuple[bool, Dict[str, int], Dict[str, int]]:
    counts = count_supervision_pixels(patch_clp, patch_cer, patch_cot, patch_cth)
    thresholds = get_patch_supervision_thresholds(mode, patch_size)
    keep = (
        counts["valid_label_pixels"] >= thresholds["min_valid_label_pixels"]
        and counts["valid_cloudy_pixels"] >= thresholds["min_valid_cloudy_pixels"]
    )
    return keep, counts, thresholds
