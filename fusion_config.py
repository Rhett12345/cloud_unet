"""
fusion_config.py
================
GPM → AGRI 融合配置。
与 config.py 分离，以便在不改动主 config 的前提下调整融合参数。
所有参数均可被环境变量覆盖（见末尾）。
"""

import os
import multiprocessing as _mp

# ─────────────────────────────────────────────────────────────────────────────
# 时间匹配（分钟）
# ─────────────────────────────────────────────────────────────────────────────
TIME_MAX_MIN = float(os.environ.get("FUSION_TIME_MAX_MIN", "10.0"))
# GPM 半小时文件与 AGRI 景的最大时间差

# ─────────────────────────────────────────────────────────────────────────────
# 空间匹配
# ─────────────────────────────────────────────────────────────────────────────
GPM_GRID_RES_DEG = float(os.environ.get("FUSION_GPM_GRID_RES_DEG", "0.1"))

# AGRI 全圆盘边缘收缩度数：避免边缘像元插值伪影
AGRI_DISK_MARGIN_DEG = float(os.environ.get("FUSION_AGRI_DISK_MARGIN_DEG", "5.0"))
AGRI_SUB_LON = float(os.environ.get("FUSION_AGRI_SUB_LON", "104.7"))

# ─────────────────────────────────────────────────────────────────────────────
# 质量控制
# ─────────────────────────────────────────────────────────────────────────────
MIN_PRECIP_QUALITY = float(os.environ.get("FUSION_MIN_PRECIP_QUALITY", "0.0"))

# GPM 格点采样步长（每隔 N 个格点采样一个，1=全采样）
GPM_SAMPLE_STEP = int(os.environ.get("FUSION_GPM_SAMPLE_STEP", "5"))

# 每景最多采样数（0=不限制）
MAX_SAMPLES_PER_SCENE = int(os.environ.get("FUSION_MAX_SAMPLES_PER_SCENE", "3000"))

# ─────────────────────────────────────────────────────────────────────────────
# 空间区域过滤（GPM 格点经纬度限制）
# 默认：赤道两侧 ~25°×25°（~2800km×2800km），覆盖南北半球，
# 位于 FY-4A (104.7°E) 与 FY-4B (~105°E) 共同覆盖范围内。
# 设空字符串 "" 或 "none" 关闭区域过滤。
# ─────────────────────────────────────────────────────────────────────────────
def _parse_region_env(name, default):
    raw = os.environ.get(name, "").strip()
    if raw.lower() in {"", "none", "false", "no"}:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

REGION_LAT_MIN = _parse_region_env("FUSION_REGION_LAT_MIN", -10.0)
REGION_LAT_MAX = _parse_region_env("FUSION_REGION_LAT_MAX",  20.0)
REGION_LON_MIN = _parse_region_env("FUSION_REGION_LON_MIN", 100.0)
REGION_LON_MAX = _parse_region_env("FUSION_REGION_LON_MAX", 130.0)

# GPM 完整覆盖：区域内 NaN 占比超过此阈值则跳过该 GPM 文件
GPM_COVERAGE_MAX_NAN_FRAC = float(os.environ.get("FUSION_GPM_COVERAGE_MAX_NAN_FRAC", "0.05"))

# ─────────────────────────────────────────────────────────────────────────────
# 多进程
# ─────────────────────────────────────────────────────────────────────────────
N_FUSION_WORKERS = int(os.environ.get("FUSION_N_WORKERS", "4" ))

# ─────────────────────────────────────────────────────────────────────────────
# 调试 / 日志
# ─────────────────────────────────────────────────────────────────────────────
FUSION_LOG_PIXEL_STATS = os.environ.get("FUSION_LOG_PIXEL_STATS", "1") == "1"

_qc_diag_raw = os.environ.get(
    "ENABLE_QC_DIAGNOSTICS",
    os.environ.get("FUSION_ENABLE_QC_DIAGNOSTICS", "0"),
)
ENABLE_QC_DIAGNOSTICS = _qc_diag_raw.strip().lower() in {"1", "true", "yes", "on"}
QC_DIAGNOSTICS_DIR = os.environ.get("FUSION_QC_DIAGNOSTICS_DIR", "runs/qc_diagnostics_gpm")
