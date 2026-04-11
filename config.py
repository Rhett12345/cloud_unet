"""
config.py
=========
Global configuration for the AGRI + MYD06 cloud property retrieval pipeline.
All paths, hyper-parameters and flags live here.
Edit this file ONLY – every other script imports from it.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Root directories
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path("/home/yuq/cloudmask/unet_workdir")           # Change to your project root

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raw data – folder-per-day layout
#     AGRI_ROOT/<YYYYMMDD>/*.HDF   (FY-4B AGRI FDI / GEO L1 files)
#     MODIS_ROOT/<YYYYMMDD>/*.hdf  (MYD06 cloud product files)
# ─────────────────────────────────────────────────────────────────────────────
AGRI_ROOT  = Path("/data/Data_yuq/FY4A/")          # parent directory of day-folders
MODIS_ROOT = Path("/data/Data_yuq/MYD06/")         # parent directory of day-folders

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
# AGRI FDI – thermal IR channel indices to use (0-based in the sorted list)
# Channels 9-15 (indices 8-14) are the thermal bands
# AGRI_BT_CHANNEL_INDICES = [8, 9, 10, 11, 12, 13, 14]   # 7 thermal channels of FY4B
AGRI_BT_CHANNEL_INDICES = [7, 8, 9, 10, 11, 12, 13]   # 7 thermal channels of FY4A

# AGRI pixel size in degrees (approx) for spatial matching
AGRI_PIXEL_DEG = 0.04

# MYD06 variables to read and store as labels
# (dataset name inside the HDF4 file)
MODIS_VARS = {
    "CLP": "Cloud_Phase_Optical_Properties",   # cloud phase  [0-6]
    "CER": "Cloud_Effective_Radius",           # µm  ×100 stored as int
    "COT": "Cloud_Optical_Thickness",          # ×100 stored as int
    "CTH": "Cloud_Top_Height",                 # m
}

# Scale factors applied AFTER reading raw integer values
MODIS_SCALE = {
    "CLP": 1.0,
    "CER": 0.01,     # integer → µm
    "COT": 0.01,
    "CTH": 1.0,      # already in metres
}

# MYD06 QC / Phase mapping  → merged to 5 classes (0=clear,1=water,2=supercool,3=mix,4=ice)
# MYD06 Cloud_Phase_Optical_Properties: 0=unknown,1=ice,2=water,3=mixed,4=ice,5=undetermined,6=bad
# MODIS_PHASE_MAP = {0: 0, 1: 4, 2: 1, 3: 3, 4: 4, 5: 0, 6: 0}
MODIS_PHASE_MAP = {0: -1, 1: 4, 2: 1, 3: 3, 4: 4, 5: -1, 6: -1}

# Maximum spatial distance (km) allowed when snapping MODIS pixel to AGRI pixel
MAX_MATCH_DIST_KM = 10.0

# Maximum time difference (minutes) between AGRI scan and MYD06 granule
MAX_TIME_DIFF_MIN = 15

# Angle filters
MAX_VZA_DEG = 65
MAX_SZA_DEG = 180   # day-only mode 65; set to 180 to include night
# 新增：分类/回归是否强依赖几何过滤
CLP_USE_GEO_FILTER = False
REG_USE_GEO_FILTER = False

# 新增：一个 patch 至少要有多少个监督像元才参与训练
MIN_PATCH_LABEL_PIXELS = 16
# ─────────────────────────────────────────────────────────────────────────────
# 7.  Patch / dataset parameters
# ─────────────────────────────────────────────────────────────────────────────
PATCH_SIZE    = (32, 32)
PATCH_OVERLAP = 16          # pixels overlap used in inference sliding window

# Train / val / test date split  (folder names, YYYYMMDD)
# Leave empty lists to use ALL available days in each split dir.
TRAIN_DATES = [
    "20190105", "20190115",
    "20190205", "20190215",
    # "20190305", "20190315",
    # "20190405", "20190415",
    # "20190505", "20190515",
    # "20190605", "20190615",
    # "20190705", "20190715",
    # "20190805", "20190815",
    # "20190905", "20190915",
    # "20191005", "20191015",
    # "20191105", "20191115",
    # "20191205", "20191215",
]   # e.g. ["20230601", "20230602", ...]
VAL_DATES   = [
    "20190125",
    # "20190325",
    # "20190525",
    # "20190725",
    # "20190925",
    # "20191125",
]
TEST_DATES  = [
    "20190225",
    # "20190425",
    # "20190625",
    # "20190825",
    # "20191025",
    # "20191225",
]

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Model hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
# Network input: only AGRI BT (no GIIRS)
AGRI_CHANNELS  = len(AGRI_BT_CHANNEL_INDICES)   # 7
GIIRS_CHANNELS = 0                               # not used

CLP_CLASSES   = 5
COMP_CHANNELS = 3   # CER, COT, CTH

MODEL_BASE_CHANNELS = 16
TRANSFORMER_DEPTH   = 2
TRANSFORMER_HEADS   = 8
TRANSFORMER_MLP_DIM = 256

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Training hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 50        # 原来 10，参考代码用 50，10 epoch 模型远未收敛
LEARNING_RATE = 1e-4
LR_PATIENCE   = 8         # 原来 5，给模型更多时间跨越平台期再降 LR
LR_FACTOR     = 0.5
MIN_LR        = 1e-6
GRAD_CLIP     = 1.0
NUM_WORKERS   = 4

# Loss weights  (CLP_CE + w_cer*CER + w_cot*COT + w_cth*CTH)
# 原来 CLP=0.5 被回归任务淹没，改为等权；待模型收敛后可再调整
LOSS_W_CLP = 1.0
LOSS_W_CER = 1.0
LOSS_W_COT = 1.0
LOSS_W_CTH = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 10. Checkpoint naming
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME     = "HIR_COMP_UNet_AGRIonly"
CHECKPOINT_BEST = MODEL_DIR / f"{MODEL_NAME}_best.pth"
CHECKPOINT_LAST = MODEL_DIR / f"{MODEL_NAME}_last.pth"

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