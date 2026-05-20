"""
config.py
=========
Global configuration for AGRI → GPM precipitation classification.

All paths, hyper-parameters and flags live here.
Edit this file ONLY – every other script imports from it.
"""

import os
from pathlib import Path


def _env_list(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    raw = raw.strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return Path(default)
    return Path(raw).expanduser()

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Root directories
# ─────────────────────────────────────────────────────────────────────────────
ROOT = _env_path("UNET_WORKDIR", "/data/Data_yuq/unet_workdir")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raw data paths
# ─────────────────────────────────────────────────────────────────────────────
AGRI_ROOT    = Path("/data/Data_yuq/FY4A/")       # FY-4A/B AGRI L1B FDI+GEO day-folders
GPM_ROOT     = Path("/data/Data_yuq/GPM_2019/")    # GPM IMERG V07B day-folders

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Paired HDF output (produced by data_fusion.py)
# ─────────────────────────────────────────────────────────────────────────────
PAIRED_ROOT      = ROOT / "paired"
PAIRED_TRAIN_DIR = PAIRED_ROOT / "train"
PAIRED_VAL_DIR   = PAIRED_ROOT / "val"
PAIRED_TEST_DIR  = PAIRED_ROOT / "test"

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Normalisation statistics cache
# ─────────────────────────────────────────────────────────────────────────────
STATS_DIR  = ROOT / "stats"
STATS_FILE = STATS_DIR / "norm_stats.npz"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model checkpoints and logs
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = ROOT / "model"
LOG_DIR   = ROOT / "logs"

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Inference output
# ─────────────────────────────────────────────────────────────────────────────
RETRIEVAL_DIR = ROOT / "retrieval"

# ─────────────────────────────────────────────────────────────────────────────
# 6.  AGRI channel selection (0-based physical channel indices)
#     FY-4A: 14 channels, FY-4B: 15 channels
#     A01→B01(0.47μm), A02→B02(0.65μm), A03→B03(0.825μm)
#     A09→B09(6.25μm), A10→B10(6.95μm)
#     A12→B13(10.8μm), A13→B14(12.0μm)
# ─────────────────────────────────────────────────────────────────────────────
AGRI_PHYSICAL_CHANNELS_A = [1, 2, 3, 9, 10, 12, 13]   # 0-based: [0,1,2,8,9,11,12]
AGRI_PHYSICAL_CHANNELS_B = [1, 2, 3, 9, 10, 13, 14]   # 0-based: [0,1,2,8,9,12,13]
AGRI_PHYSICAL_CHANNELS = AGRI_PHYSICAL_CHANNELS_A       # default A, auto-detect at runtime

# Legacy alias for backward compatibility
AGRI_BT_CHANNEL_INDICES = [c - 1 for c in AGRI_PHYSICAL_CHANNELS]  # 0-based

# Visible channel indices within the 7-channel stack (0-based): A01, A02, A03
VIS_CHANNEL_INDICES = [0, 1, 2]

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Precipitation classification
# ─────────────────────────────────────────────────────────────────────────────
# Class 0: No-rain        precip < 0.1 mm/h
# Class 1: Light rain      0.1 ≤ precip < 2.5 mm/h
# Class 2: Moderate rain   2.5 ≤ precip < 8.0 mm/h
# Class 3: Heavy rain      precip ≥ 8.0 mm/h
PRECIP_THRESHOLDS = [0.0, 0.1, 2.5, 8.0]   # lower bounds for each class
PRECIP_CLASS_NAMES = ["No-rain", "Light rain", "Moderate rain", "Heavy rain"]
PRECIP_CLASSES = 4

def precip_to_class(precip_mmh):
    """Convert precipitation rate (mm/h) to class label 0-3."""
    if precip_mmh < 0.1:
        return 0
    elif precip_mmh < 2.5:
        return 1
    elif precip_mmh < 8.0:
        return 2
    else:
        return 3

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Data fusion parameters
# ─────────────────────────────────────────────────────────────────────────────
GPM_TIME_MAX_MIN = float(os.environ.get("GPM_TIME_MAX_MIN", "15.0"))
# Maximum time difference (minutes) between AGRI scan and GPM half-hour file

GPM_GRID_RES_DEG = 0.1       # GPM IMERG grid resolution in degrees
GPM_LON_SIZE = 3600           # GPM longitude dimension
GPM_LAT_SIZE = 1800           # GPM latitude dimension
GPM_FILL_VALUE = -9999.9      # GPM IMERG fill value

# AGRI pixel size in degrees (approx) for resampling
AGRI_PIXEL_DEG = 0.04

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Patch / dataset parameters
# ─────────────────────────────────────────────────────────────────────────────
PATCH_SIZE    = (11, 11)
PATCH_OVERLAP = 5             # pixels overlap for inference sliding window

# ─────────────────────────────────────────────────────────────────────────────
# 10. Train / val / test date split
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_DATES = _env_list("UNET_TRAIN_DATES", [
    "20190101", "20190102", "20190103", "20190104", "20190105",
    "20190106", "20190107", "20190108", "20190109", "20190110",
    "20190111", "20190112",
])
VAL_DATES   = _env_list("UNET_VAL_DATES", [
    "20190113", "20190114", "20190115",
])
TEST_DATES  = _env_list("UNET_TEST_DATES", [
    "20190116", "20190117", "20190118", "20190119",
])

# ─────────────────────────────────────────────────────────────────────────────
# 11. Model hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
AGRI_CHANNELS  = len(AGRI_PHYSICAL_CHANNELS)   # 7
GEO_CHANNELS   = 4                              # lat, lon, VZA, SZA (kept for compat)
NUM_CLASSES    = PRECIP_CLASSES                 # 4

UNET_BASE_CHANNELS = 64

# ─────────────────────────────────────────────────────────────────────────────
# 12. Training hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
LR_PATIENCE   = 6
LR_FACTOR     = 0.5
MIN_LR        = 1e-6
GRAD_CLIP     = 1.0
NUM_WORKERS   = 5

# Early stopping
EARLY_STOP_PATIENCE = 10

# Loss
LOSS_TYPE = os.environ.get("UNET_LOSS_TYPE", "weighted_ce")  # "weighted_ce" | "focal"

# Sample quality filter
SAMPLE_QUALITY_FILTER_ENABLED = _env_bool("UNET_SAMPLE_QUALITY_FILTER", False)

CHECKPOINT_MONITOR = os.environ.get("UNET_CHECKPOINT_MONITOR", "val_f1_class3")

# ─────────────────────────────────────────────────────────────────────────────
# 13. Checkpoint naming
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME     = "AGRI_GPM_Precip_UNet"
CHECKPOINT_BEST = MODEL_DIR / f"{MODEL_NAME}_best.pth"
CHECKPOINT_LAST = MODEL_DIR / f"{MODEL_NAME}_last.pth"
CHECKPOINT_BEST_LOSS = MODEL_DIR / f"{MODEL_NAME}_best_loss.pth"
CHECKPOINT_BEST_OA = MODEL_DIR / f"{MODEL_NAME}_best_oa.pth"
CHECKPOINT_BEST_F1_C3 = MODEL_DIR / f"{MODEL_NAME}_best_f1_c3.pth"

# ─────────────────────────────────────────────────────────────────────────────
# 14. Evaluation / inference
# ─────────────────────────────────────────────────────────────────────────────
EVAL_OUTPUT_DIR = ROOT / "eval_gpm"

# ─────────────────────────────────────────────────────────────────────────────
# 15. Misc
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
LOG_LEVEL   = "INFO"

# Ensure output directories exist at import time
for _d in [PAIRED_TRAIN_DIR, PAIRED_VAL_DIR, PAIRED_TEST_DIR,
           STATS_DIR, MODEL_DIR, LOG_DIR, RETRIEVAL_DIR, EVAL_OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
