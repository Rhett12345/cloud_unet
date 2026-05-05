"""
fusion_config.py
================
MODIS→AGRI 单像元最近邻融合配置。
与 config.py 分离，以便在不改动主 config 的前提下调整融合参数。
所有参数均可被环境变量覆盖（见末尾）。
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# 时间匹配（分钟）
# ─────────────────────────────────────────────────────────────────────────────
TIME_HIGH_Q_MIN   = float(os.environ.get("FUSION_TIME_HIGH_Q_MIN",   "5.0"))
TIME_LOW_Q_MIN    = float(os.environ.get("FUSION_TIME_LOW_Q_MIN",    "7.5"))
# CLP 分类标签：dt <= TIME_LOW_Q_MIN
# 回归标签 REG_TIME_MAX_MIN：更严格的时间上限（默认 5.0 min）

REG_TIME_MAX_MIN  = float(os.environ.get("FUSION_REG_TIME_MAX_MIN",  "7.5"))
# 与 TIME_LOW_Q_MIN 统一为 7.5 min，参照 GeoISCLD-Net 简化 QC 链

SCAN_TIME_FALLBACK_WEIGHT = float(os.environ.get("FUSION_SCAN_TIME_FALLBACK_WEIGHT", "0.7"))
# 当无法获取像元级 scan_time 时，样本权重乘以此系数

# ─────────────────────────────────────────────────────────────────────────────
# 严格模式开关
# ─────────────────────────────────────────────────────────────────────────────
REQUIRE_MYD03_1KM = os.environ.get("FUSION_REQUIRE_MYD03_1KM", "1") == "1"
# 要求 MYD03 1km 地理定位，避免退化为 MYD06 5km 坐标上采样

REQUIRE_SCAN_TIME = os.environ.get("FUSION_REQUIRE_SCAN_TIME", "1") == "1"
# 要求逐像元扫描时间，避免 MATCH_DT_MIN 退化为文件名级时间差

# ─────────────────────────────────────────────────────────────────────────────
# 空间匹配
# ─────────────────────────────────────────────────────────────────────────────
AGRI_SEARCH_RADIUS_KM = float(os.environ.get("FUSION_AGRI_SEARCH_RADIUS_KM", "2.5"))
# KD-tree 搜索半径 (km)

AGRI_DISK_MARGIN_DEG = float(os.environ.get("FUSION_AGRI_DISK_MARGIN_DEG", "5.0"))
# AGRI 全圆盘边缘向内收缩度数：仅保留距离圆盘边界 ≥ 此值的内部像元，
# 避免 MODIS 条带与 AGRI 圆盘边缘不完整重叠导致的低质量匹配和计算浪费。

AGRI_SUB_LON = float(os.environ.get("FUSION_AGRI_SUB_LON", "104.7"))
# FY-4A 星下点经度（用于计算像元到圆盘中心的角距离）。

# ─────────────────────────────────────────────────────────────────────────────
# 视差修正参数（占位，留用于未来实现）
# ─────────────────────────────────────────────────────────────────────────────
PARALLAX_HIGH_CTH_M   = float(os.environ.get("FUSION_PARALLAX_HIGH_CTH_M",   "6000.0"))
PARALLAX_HIGH_VZA_DEG = float(os.environ.get("FUSION_PARALLAX_HIGH_VZA_DEG", "40.0"))

# ─────────────────────────────────────────────────────────────────────────────
# 多进程
# ─────────────────────────────────────────────────────────────────────────────
import multiprocessing as _mp
N_FUSION_WORKERS = int(os.environ.get("FUSION_N_WORKERS",
                        # str(max(1, (_mp.cpu_count() or 4) - 1))
                            str(16)
                                       ))

# ─────────────────────────────────────────────────────────────────────────────
# 调试 / 日志
# ─────────────────────────────────────────────────────────────────────────────
FUSION_LOG_PIXEL_STATS  = os.environ.get("FUSION_LOG_PIXEL_STATS", "1") == "1"

_qc_diag_raw = os.environ.get(
    "ENABLE_QC_DIAGNOSTICS",
    os.environ.get("FUSION_ENABLE_QC_DIAGNOSTICS", "0"),
)
ENABLE_QC_DIAGNOSTICS = _qc_diag_raw.strip().lower() in {"1", "true", "yes", "on"}
QC_DIAGNOSTICS_DIR = os.environ.get("FUSION_QC_DIAGNOSTICS_DIR", "runs/qc_diagnostics")
