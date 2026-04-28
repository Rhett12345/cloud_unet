"""
fusion_config.py
================
质量优先融合专用配置。
与 config.py 分离，以便在不改动主 config 的前提下调整融合参数。
所有参数均可被环境变量覆盖（见末尾）。
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# 时间匹配分层阈值（分钟）
# ─────────────────────────────────────────────────────────────────────────────
TIME_HIGH_Q_MIN   = float(os.environ.get("FUSION_TIME_HIGH_Q_MIN",   "5.0"))   # <= 5 min：权重 1.0
TIME_LOW_Q_MIN    = float(os.environ.get("FUSION_TIME_LOW_Q_MIN",    "7.5"))   # 5~7.5：降权; >7.5 丢弃
SCAN_TIME_FALLBACK_WEIGHT = float(os.environ.get("FUSION_SCAN_TIME_FALLBACK_WEIGHT", "0.7"))
# 当无法获取像元级 scan_time 时，文件名时间作 fallback，权重乘以此系数
REQUIRE_MYD03_1KM = os.environ.get("FUSION_REQUIRE_MYD03_1KM", "1") == "1"
# 严格融合默认要求 MYD03 1km geolocation，避免静默退回 MYD06 5km 经纬度重复上采样。
REQUIRE_SCAN_TIME = os.environ.get("FUSION_REQUIRE_SCAN_TIME", "1") == "1"
# 严格融合默认要求逐扫描/像元时间，避免 MATCH_DT_MIN 退化为文件名级时间差。

# ─────────────────────────────────────────────────────────────────────────────
# 空间聚合参数
# ─────────────────────────────────────────────────────────────────────────────
# AGRI 4km footprint 搜索半径 (km)。设为 2.5km 以减少 MODIS 混合像元引入的标签噪声。
AGRI_SEARCH_RADIUS_KM  = float(os.environ.get("FUSION_AGRI_SEARCH_RADIUS_KM", "2.5"))

# 期望落入一个 AGRI 4km 像元的 MYD06 1km / 5km 像元数
# 面积比: π*(2.5)² ≈ 19.6 km²，约 20 个 1km 像元，保留最近 12 个
EXPECTED_1KM_PER_AGRI  = float(os.environ.get("FUSION_EXPECTED_1KM_PER_AGRI",  "12.0"))
EXPECTED_5KM_PER_AGRI  = float(os.environ.get("FUSION_EXPECTED_5KM_PER_AGRI",  "1.0"))

# ─────────────────────────────────────────────────────────────────────────────
# 质量控制阈值（质量优先：宁可丢弃，不要伪标签）
# ─────────────────────────────────────────────────────────────────────────────
MIN_VALID_PIX           = int(os.environ.get("FUSION_MIN_VALID_PIX",           "1"))
OVERLAP_FRAC_MIN        = float(os.environ.get("FUSION_OVERLAP_FRAC_MIN",      "0.5"))
CLOUD_FRAC_MIN_CLOUDY   = float(os.environ.get("FUSION_CLOUD_FRAC_MIN_CLOUDY", "0.6"))
PHASE_CONSISTENCY_MIN   = float(os.environ.get("FUSION_PHASE_CONSISTENCY_MIN", "0.7"))

# 纯云模式开关：仅保留 cloud_fraction > PURE_CLOUD_FRAC 的 COT/CER 标签
PURE_CLOUD_ONLY         = os.environ.get("FUSION_PURE_CLOUD_ONLY", "0") == "1"
PURE_CLOUD_FRAC         = float(os.environ.get("FUSION_PURE_CLOUD_FRAC",       "0.9"))

# 回归标签最终门控比 CLP 更严格；分类可保留边界/clear 样本，CER/COT/CTH 宁可少而准。
REG_TIME_MAX_MIN        = float(os.environ.get("FUSION_REG_TIME_MAX_MIN",        "3.0"))
REG_OVERLAP_FRAC_MIN    = float(os.environ.get("FUSION_REG_OVERLAP_FRAC_MIN",    str(OVERLAP_FRAC_MIN)))
REG_CLOUD_FRAC_MIN      = float(os.environ.get("FUSION_REG_CLOUD_FRAC_MIN",      str(CLOUD_FRAC_MIN_CLOUDY)))
REG_PHASE_CONSISTENCY_MIN = float(os.environ.get("FUSION_REG_PHASE_CONSISTENCY_MIN", "0.8"))

# COT 对数域 epsilon（避免 log(0)）
COT_LOG_EPS             = 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# 视差修正参数（当前为占位，留用于未来实现）
# ─────────────────────────────────────────────────────────────────────────────
PARALLAX_HIGH_CTH_M     = float(os.environ.get("FUSION_PARALLAX_HIGH_CTH_M",   "6000.0"))
PARALLAX_HIGH_VZA_DEG   = float(os.environ.get("FUSION_PARALLAX_HIGH_VZA_DEG", "40.0"))
FY4_SAT_LON_DEG         = float(os.environ.get("FUSION_FY4_SAT_LON_DEG",       "104.7"))

# ─────────────────────────────────────────────────────────────────────────────
# 多进程参数
# ─────────────────────────────────────────────────────────────────────────────
import os as _os, multiprocessing as _mp
N_FUSION_WORKERS = int(_os.environ.get("FUSION_N_WORKERS",
                        # str(max(1, (_mp.cpu_count() or 4) - 1))
                            str(16)
                                       ))
# 每个 AGRI 文件在子进程中独立处理；主进程只做调度和写盘

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
