"""
config.py
=========
Global configuration for the AGRI + MYD06 cloud property retrieval pipeline.
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
# ROOT = Path("/home/yuq/cloudmask/unet/unet_workdir")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raw data – folder-per-day layout
#     AGRI_ROOT/<YYYYMMDD>/*.HDF   (FY-4B AGRI FDI / GEO L1 files)
#     MODIS_ROOT/<YYYYMMDD>/*.hdf  (MYD06 cloud product files)
#     MYD03_ROOT/<YYYYMMDD>/*.hdf  (MYD03 1km geolocation files)
# ─────────────────────────────────────────────────────────────────────────────
AGRI_ROOT    = Path("/data/Data_yuq/FY4A/")          # parent directory of day-folders (L1B FDI+GEO)
FY4A_L2_ROOT = Path("/data/Data_yuq/FY4A_L2/")     # parent directory of L2 CLP/CTH day-folders
MODIS_ROOT   = Path("/data/Data_yuq/MYD06/")         # parent directory of day-folders
MYD03_ROOT   = Path("/data/Data_yuq/MYD03/")         # parent directory of day-folders

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
STATS_FILE = STATS_DIR / "norm_stats.npz"   # saved by compute_stats.py

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
# 6.  Data fusion parameters
# ─────────────────────────────────────────────────────────────────────────────
# AGRI FDI – selected channel indices to use (0-based in the sorted list)
# 当前输入为：ch2(0.65µm) + ch5(1.61µm) + ch8-14(IR)
# 保留变量名 AGRI_BT_CHANNEL_INDICES 以兼容现有训练/推理代码。
# FY-4A: 14通道，索引 8-13 = Ch9(6.25)-Ch14(13.5)
# FY-4B: 15通道，B11(7.42)无A星对应 → 跳过B11，索引 [8,9,11,12,13,14]
_AGRI_BT_CHANNEL_INDICES_A = [8, 9, 10, 11, 12, 13]
_AGRI_BT_CHANNEL_INDICES_B = [8, 9, 11, 12, 13, 14]
AGRI_BT_CHANNEL_INDICES = _AGRI_BT_CHANNEL_INDICES_A  # 默认A星，运行时自动检测

# AGRI pixel size in degrees (approx) for spatial matching
AGRI_PIXEL_DEG = 0.04

# MYD06 primary supervision variables (all 1km SDS names)
# CER/COT use _16 (1.6 µm): lower coverage (~24.6%) but better BT-CER correlation
# and higher model prediction accuracy (CLP OA 48.6% vs 45.0% with combined).
# Uncertainty SDS: Cloud_Effective_Radius_Uncertainty_16 / Cloud_Optical_Thickness_Uncertainty_16
MODIS_VARS = {
    "CLP": "Cloud_Phase_Infrared_1km",     # IR phase: clear / water / ice / undetermined
    # "CER": "Cloud_Effective_Radius_16",  # removed: focus on CLP+CTH only
    # "COT": "Cloud_Optical_Thickness_16", # removed: focus on CLP+CTH only
    "CTH": "cloud_top_height_1km",         # m
}

# Auxiliary MYD06 SDS used for quality control only
MODIS_QC_VARS = {
    "CLP_OPT": "Cloud_Phase_Optical_Properties",
    "CTP": "cloud_top_pressure_1km",
    "CTT": "cloud_top_temperature_1km",
    "CTM": "cloud_top_method_1km",
}

# Scale factors applied AFTER reading raw integer values
MODIS_SCALE = {
    "CLP": 1.0,
    # "CER": 0.01,     # removed: focus on CLP+CTH only
    # "COT": 0.01,     # removed: focus on CLP+CTH only
    "CTH": 1.0,      # already in metres
}

# IR phase → training phase space (0=clear, 1=water, 2=ice)
# 临时三分类设置：Cloud_Phase_Infrared_1km 中未稳定覆盖的 supercool/mixed
# 不作为单独训练类别；旧融合文件中的 ice=4 会在 dataset.py 中重映射为 ice=2。
CLP_CLASS_NAMES = ["Clear", "Water", "Ice"]
CLP_LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 2}
MODIS_PHASE_MAP = {0: 0, 1: 1, 2: 2, 3: -1, 4: -1, 5: -1, 6: -1}

# AGRI L2 Cloud Phase → training phase mapping (5 → 3 classes)
# AGRI L2 values: 0=Clear, 1=Water, 2=Supercooled, 3=Mixed, 4=Ice, 126=Space, 127=Fill
# Supercooled (2) → Water (1): physically liquid water droplets
# Mixed (3) → Ice (2): mixed-phase clouds treated as ice-like
AGRI_L2_CLP_PHASE_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
AGRI_L2_CTH_VALID_RANGE = (1.0, 20000.0)  # from product metadata valid_range
AGRI_L2_CTH_FILL_VALUE  = -999.0

# Optical phase (QC only) mapped to the same phase space when需要一致性检查。
MODIS_OPTICAL_PHASE_MAP = {0: -1, 1: 0, 2: 1, 3: 2, 4: -1}

# Maximum spatial distance (km) allowed when snapping MODIS pixel to AGRI pixel
MAX_MATCH_DIST_KM = 3.0

# Maximum time difference (minutes) between AGRI scan and MYD06 granule
MAX_TIME_DIFF_MIN = 5

# Angle filters  (CLP 放宽以保留更多分类监督；回归保持严格)
MAX_VZA_DEG     = _env_float("UNET_MAX_VZA_DEG", 65)
MAX_SZA_DEG     = _env_float("UNET_MAX_SZA_DEG", 65)
MAX_VZA_DEG_CLP = _env_float("UNET_MAX_VZA_DEG_CLP", 65)
MAX_SZA_DEG_CLP = _env_float("UNET_MAX_SZA_DEG_CLP", 65)
MAX_CTH_M       = _env_float("UNET_MAX_CTH_M", 18000)
# 分类/回归是否强依赖几何过滤
CLP_USE_GEO_FILTER = True
REG_USE_GEO_FILTER = True

# Patch 监督样本过滤规则（train / val / test 共用同一套默认策略）。
# 最终门槛 = max(最少像元数, patch_area * 最小占比)。
# 32×32 patch 下，默认要求：
#   - 足够的有效 CLP 监督像元
#   - 不再强制要求有云回归像元，避免 clear-dominant patch 被系统性丢弃
# 回归监督缺失时训练 loss 会自动 mask，不影响 CLP 分类学习。
PATCH_FILTER_RULES = {
    "default": {
        "min_valid_label_pixels": 1024,
        "min_valid_label_ratio": 0.25,
        "min_valid_cloudy_pixels": 300,
        "min_valid_cloudy_ratio": 0.08,
    },
    "train": {},
    "val": {
        "min_valid_label_pixels": 600,
        "min_valid_label_ratio": 0.15,
        "min_valid_cloudy_pixels": 200,
        "min_valid_cloudy_ratio": 0.05,
    },
    "test": {
        "min_valid_label_pixels": 600,
        "min_valid_label_ratio": 0.15,
        "min_valid_cloudy_pixels": 200,
        "min_valid_cloudy_ratio": 0.05,
    },
}

# 融合输出只保留有监督样本，不再把整幅全圆盘直接写入训练 HDF5。
FUSION_OUTPUT_MODE = "samples_only"   # "samples_only" | "full_disk"

# 安全写入：先写临时 HDF5，校验通过后再转正。
TEMP_H5_SUFFIX = ".tmp.h5"
KEEP_TEMP_H5_ON_ERROR = True

# 弱质量 MYD06 样本在最早阶段直接过滤，不进入匹配、统计和 patch 采样。
# 参照 GeoISCLD-Net：关闭 Cloud_Mask 比特解码、光学相态、CTH 辅助等复杂过滤，
# 只保留值域 + 时间窗口 + VZA/SZA 几何限制。
MODIS_FILTER_WEAK_QUALITY = False

# Cloud_Mask cloudiness: 0=Confident Cloudy, 1=Probably Cloudy,
#                        2=Probably Clear,  3=Confident Clear
# CLP 是分类监督，保留所有状态有效的 cloud mask 像元；
# CER/COT/CTH 是回归监督，仍只保留高置信度 cloudy / clear 以控制噪声。
MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_CLP = (0, 1, 2, 3)
MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG = (0, 3)
# Backward-compatible aliases for older code paths.
MODIS_ALLOWED_CLOUD_MASK_FLAGS_1KM = MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG
MODIS_ALLOWED_CLOUD_MASK_FLAGS_5KM = MODIS_ALLOWED_CLOUD_MASK_FLAGS_FOR_REG

# 光学厚度 / 有效半径不确定度过大时视为弱质量。
# 0~200% 是产品公开范围；默认 80% 能过滤明显不稳定的检索结果，
# 但又不会像 30%/50% 那样过于激进。
MODIS_MAX_COT_UNCERTAINTY_PCT = 100.0
MODIS_MAX_CER_UNCERTAINTY_PCT = 100.0

# Optical-property retrieval QC：仅保留 Cloud_Phase_Optical_Properties 指示为云的像元
# 1=water cloud, 2=ice cloud, 3/4=undetermined
MODIS_ALLOWED_OPTICAL_PHASES_FOR_COP = (1, 2, 3, 4)
MODIS_REQUIRE_OPTICAL_PHASE_FOR_COP = False

# 可选：要求 IR phase 与 optical phase 在可比时一致；默认关闭，避免样本过度收缩。
MODIS_REQUIRE_PHASE_AGREEMENT = False

# Cloud-top auxiliary QC（1km）：method 1/2/3/4 为 CO2-slicing，6 为 IR window。
MODIS_ALLOWED_CLOUD_TOP_METHODS = (1, 2, 3, 4, 6)
MODIS_REQUIRE_CTH_AUX = False
# ─────────────────────────────────────────────────────────────────────────────
# 7.  Patch / dataset parameters
# ─────────────────────────────────────────────────────────────────────────────
PATCH_SIZE    = (64, 64)
PATCH_OVERLAP = 32          # pixels overlap used in inference sliding window

# Train / val / test date split  (folder names, YYYYMMDD)
# Leave empty lists to use ALL available days in each split dir.
# 2019-04-01 ~ 2019-05-31 (61 days), split by day-of-month % 7:
#   Train: dom%7 in {0,1,3,4,6} (~43 days)
#   Val:   dom%7 == 2            (~9 days)
#   Test:  dom%7 == 5            (~9 days)
TRAIN_DATES = _env_list("UNET_TRAIN_DATES", [
    # April (~11 days, every other from the full 21-day train set)
    "20190401", "20190404", "20190407", "20190410", "20190413",
    "20190415", "20190418", "20190421", "20190424", "20190427",
    "20190429",
    # May (~11 days)
    "20190501", "20190504", "20190507", "20190510", "20190513",
    "20190515", "20190518", "20190521", "20190524", "20190527",
    "20190529", "20190531",
])
VAL_DATES   = _env_list("UNET_VAL_DATES", [
    "20190402", "20190409", "20190416", "20190423", "20190430",
    "20190502", "20190509", "20190516", "20190523", "20190530",
])
TEST_DATES  = _env_list("UNET_TEST_DATES", [
    "20190405", "20190412", "20190419", "20190426",
    "20190505", "20190512", "20190519", "20190526",
])

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Model hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
# Network input: selected AGRI channels (VIS/NIR + IR, no GIIRS)
AGRI_CHANNELS  = len(AGRI_BT_CHANNEL_INDICES)   # 8
GIIRS_CHANNELS = 0                               # not used

CLP_CLASSES   = len(CLP_CLASS_NAMES)
COMP_CHANNELS = 1   # CTH only (CER/COT removed)

UNET_BASE_CHANNELS  = 64

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Training hyper-parameters
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

# Loss weights  (CLP_CE + w_cth*CTH)
# 原来 CLP=0.5 被回归任务淹没，改为等权；待模型收敛后可再调整
LOSS_W_CLP = _env_float("UNET_LOSS_W_CLP", 1.0)
# LOSS_W_CER = _env_float("UNET_LOSS_W_CER", 1.0)  # removed
# LOSS_W_COT = _env_float("UNET_LOSS_W_COT", 1.0)  # removed
LOSS_W_CTH = _env_float("UNET_LOSS_W_CTH", 1.0)

# Optional sample-level quality gate for upper-bound experiments.
SAMPLE_QUALITY_FILTER_ENABLED = _env_bool("UNET_SAMPLE_QUALITY_FILTER", False)
QUALITY_MIN_OVERLAP_FRAC = _env_float("UNET_QUALITY_MIN_OVERLAP_FRAC", 0.0)
QUALITY_MAX_TIME_DIFF_MIN = _env_float("UNET_QUALITY_MAX_TIME_DIFF_MIN", 1e9)
QUALITY_MIN_PHASE_CONSIST = _env_float("UNET_QUALITY_MIN_PHASE_CONSIST", 0.0)
QUALITY_MIN_CLOUD_FRAC = _env_float("UNET_QUALITY_MIN_CLOUD_FRAC", 0.0)
QUALITY_MIN_VALID_CLOUDY_PIXELS = int(os.environ.get("UNET_QUALITY_MIN_VALID_CLOUDY_PIXELS", "0"))

CHECKPOINT_MONITOR = os.environ.get("UNET_CHECKPOINT_MONITOR", "val_macro_acc")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Checkpoint naming
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME     = "HIR_COMP_UNet_AGRIonly"
CHECKPOINT_BEST = MODEL_DIR / f"{MODEL_NAME}_best.pth"
CHECKPOINT_LAST = MODEL_DIR / f"{MODEL_NAME}_last.pth"
CHECKPOINT_BEST_LOSS = MODEL_DIR / f"{MODEL_NAME}_best_loss.pth"
CHECKPOINT_BEST_OA = MODEL_DIR / f"{MODEL_NAME}_best_oa.pth"
CHECKPOINT_BEST_MACRO = MODEL_DIR / f"{MODEL_NAME}_best_macro.pth"

# ─────────────────────────────────────────────────────────────────────────────
# 11. Evaluation / inference
# ─────────────────────────────────────────────────────────────────────────────
# MODIS cross-validation comparison
MODIS_EVAL_DIR    = MODIS_ROOT   # same as MODIS_ROOT by default
EVAL_OUTPUT_DIR   = ROOT / "eval"
EVAL_RESULTS_CSV  = EVAL_OUTPUT_DIR / "MODIS_vs_model_results.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 12. Misc
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
LOG_LEVEL   = "INFO"    # DEBUG / INFO / WARNING

# Ensure output directories exist at import time
for _d in [PAIRED_TRAIN_DIR, PAIRED_VAL_DIR, PAIRED_TEST_DIR,
           STATS_DIR, MODEL_DIR, LOG_DIR, RETRIEVAL_DIR, EVAL_OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
