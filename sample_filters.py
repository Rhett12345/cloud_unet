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


def _finite_float(fields: Dict[str, float], name: str) -> float:
    value = fields.get(name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def sample_passes_quality(fields: Dict[str, float]) -> bool:
    """Return whether a samples-only H5 patch passes optional quality gates."""
    if not getattr(cfg, "SAMPLE_QUALITY_FILTER_ENABLED", False):
        return True

    overlap = _finite_float(fields, "mean_overlap_frac")
    dt_max = _finite_float(fields, "max_time_diff_min")
    phase = _finite_float(fields, "mean_phase_consist")
    cloud = _finite_float(fields, "mean_cloud_frac")
    cloudy_px = _finite_float(fields, "valid_cloudy_pixels")

    checks = [
        np.isfinite(overlap) and overlap >= float(getattr(cfg, "QUALITY_MIN_OVERLAP_FRAC", 0.0)),
        np.isfinite(dt_max) and dt_max <= float(getattr(cfg, "QUALITY_MAX_TIME_DIFF_MIN", 1e9)),
        np.isfinite(phase) and phase >= float(getattr(cfg, "QUALITY_MIN_PHASE_CONSIST", 0.0)),
        np.isfinite(cloud) and cloud >= float(getattr(cfg, "QUALITY_MIN_CLOUD_FRAC", 0.0)),
        np.isfinite(cloudy_px) and cloudy_px >= int(getattr(cfg, "QUALITY_MIN_VALID_CLOUDY_PIXELS", 0)),
    ]
    return all(checks)
