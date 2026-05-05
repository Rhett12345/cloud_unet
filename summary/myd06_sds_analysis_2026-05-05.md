# MYD06 SDS 清单与数据质量分析 — 2026-05-05

## 文件概况

MYD06 产品为 HDF4 格式（非 HDF5），使用 pyhdf.SD 读取。每个 granule 包含两类分辨率的 SDS：

| 分辨率 | 尺寸 | 内容 |
|--------|------|------|
| 5 km | 406×270 | 扫描几何、部分云参数、Cloud_Mask_5km |
| 1 km | 2030×1354 | 云属性检索（CER/COT/CTH/CLP）、Cloud_Mask_1km |

5km : 1km = 1:5 像元比（每个 5km 像元对应 5×5 个 1km 像元）。

---

## 完整 SDS 清单

### 5km 网格 (406×270)

| SDS | dtype | 说明 |
|-----|-------|------|
| `Latitude` | float32 | 像元中心纬度 |
| `Longitude` | float32 | 像元中心经度 |
| `Scan_Start_Time` | float64 | 逐像元扫描时间（TAI93 epoch, seconds） |
| `Solar_Zenith` | int16 | 太阳天顶角（scaled） |
| `Solar_Zenith_Day` | int16 | 白天太阳天顶角 |
| `Solar_Zenith_Night` | int16 | 夜间太阳天顶角 |
| `Solar_Azimuth` | int16 | 太阳方位角 |
| `Sensor_Zenith` | int16 | 传感器天顶角 |
| `Sensor_Azimuth` | int16 | 传感器方位角 |
| `Brightness_Temperature` | int16 | 7 通道亮温 (7×406×270) |
| `Surface_Temperature` | int16 | 地表温度 |
| `Surface_Pressure` | int16 | 地表气压 |
| `Cloud_Height_Method` | uint8 | 云高反演方法 |
| `Cloud_Top_Height` | int16 | 云顶高度 |
| `Cloud_Top_Pressure` | int16 | 云顶气压 |
| `Cloud_Top_Temperature` | int16 | 云顶温度 |
| `Tropopause_Height` | int16 | 对流层顶高度 |
| `Cloud_Fraction` | uint8 | 云量 |
| `Cloud_Effective_Emissivity` | uint8 | 有效发射率 |
| `Cloud_Top_Pressure_Infrared` | int16 | IR 云顶气压 |
| `Spectral_Cloud_Forcing` | int16 | 光谱云强迫 (5×406×270) |
| `Cloud_Top_Pressure_From_Ratios` | int16 | 比值法云顶气压 |
| `Radiance_Variance` | int16 | 辐射方差 |
| `Cloud_Phase_Infrared` | uint8 | IR 云相态 (5km) |
| `Cloud_Phase_Infrared_Day` | uint8 | 白天 IR 相态 |
| `Cloud_Phase_Infrared_Night` | uint8 | 夜间 IR 相态 |
| `Cloud_Mask_5km` | uint8 | 5km 云掩膜 (406×270×2) |
| `Quality_Assurance_5km` | uint8 | 5km QA (406×270×10) |

### 1km 网格 (2030×1354)

| SDS | dtype | 说明 |
|-----|-------|------|
| `Cloud_Phase_Infrared_1km` | uint8 | **IR 云相态**：0=Clear, 1=Water, 2=Ice, 6=Undetermined |
| `cloud_top_height_1km` | int16 | **云顶高度** (m) |
| `cloud_top_pressure_1km` | int16 | 云顶气压 (hPa) |
| `cloud_top_temperature_1km` | int16 | 云顶温度 (K) |
| `cloud_top_method_1km` | uint8 | 云顶反演方法 |
| `cloud_emissivity_1km` | uint8 | 云发射率 |
| `cloud_emiss11_1km` ~ `cloud_emiss85_1km` | int16 | 各通道云发射率 |
| `surface_temperature_1km` | int16 | 地表温度 |
| `IRP_CTH_Consistency_Flag_1km` | uint8 | IR 相态—CTH 一致性 |
| `os_top_flag_1km` | uint8 | 单层云顶标记 |
| `Cloud_Phase_Optical_Properties` | uint8 | **光学相态**：1=Water, 2=Ice, 3/4=Undetermined |
| `Cloud_Multi_Layer_Flag` | uint8 | 多层云标记 |
| `Cirrus_Reflectance` | int16 | 卷云反射率 |
| `Cirrus_Reflectance_Flag` | uint8 | 卷云反射率标记 |
| `Cloud_Mask_1km` | uint8 | 1km 云掩膜 (2030×1354×2) |
| `Cloud_Mask_SPI` | int16 | 云掩膜 SPI |
| `Quality_Assurance_1km` | uint8 | 1km QA (2030×1354×9) |
| `IRW_Low_Cloud_Temperature_From_COP` | int16 | COP 红外窗区低温云温度 |
| `Above_Cloud_Water_Vapor_094` | int16 | 云上水汽 0.94µm |
| `Atm_Corr_Refl` | int16 | 大气校正反射率 (2030×1354×6) |
| `Retrieval_Failure_Metric` | int16 | 反演失败指标 (2030×1354×3) |

### CER/COT 变体（全部 1km）

**有效半径 (CER) 变体：**

| SDS | 说明 |
|-----|------|
| `Cloud_Effective_Radius` | 标准 2.1µm 通道 |
| `Cloud_Effective_Radius_16` | **1.6µm 通道（当前使用）** |
| `Cloud_Effective_Radius_37` | 3.7µm 通道 |
| `Cloud_Effective_Radius_1621` | 1.6+2.1µm 联合 |
| `Cloud_Effective_Radius_PCL` | PCL 后处理 |
| `Cloud_Effective_Radius_16_PCL` | 1.6µm + PCL |
| `Cloud_Effective_Radius_37_PCL` | 3.7µm + PCL |
| `Cloud_Effective_Radius_1621_PCL` | 1.6+2.1µm + PCL |

**光学厚度 (COT) 变体：** 同上 8 种命名模式。

**不确定度 (Uncertainty)：**
- `Cloud_Effective_Radius_Uncertainty` / `_16` / `_37` / `_1621`
- `Cloud_Optical_Thickness_Uncertainty` / `_16` / `_37` / `_1621`
- `Cloud_Water_Path_Uncertainty` / `_16` / `_37` / `_1621`

**云水路径 (CWP)：** 对应 8 种变体每种都有。

所有 CER/COT 共用属性：
- `scale_factor = 0.01`（整数 ÷100 → 物理值）
- `valid_range = [0, 10000]`（CER: 0–100µm, COT: 0–100）
- `_FillValue = -9999`

### LUT 表（小数组）

| SDS | shape | 说明 |
|-----|-------|------|
| `Extinction_Efficiency_Ice` | 12×7 | 冰云消光效率 LUT |
| `Asymmetry_Parameter_Ice` | 12×7 | 冰云不对称参数 LUT |
| `Single_Scatter_Albedo_Ice` | 12×7 | 冰云单次散射反照率 LUT |
| `Extinction_Efficiency_Liq` | 18×7 | 水云消光效率 LUT |
| `Asymmetry_Parameter_Liq` | 18×7 | 水云不对称参数 LUT |
| `Single_Scatter_Albedo_Liq` | 18×7 | 水云单次散射反照率 LUT |
| `Statistics_1km_sds` | 17 | 1km SDS 统计参数 |

---

## 关键数据覆盖分析

### 对比：有 CER 的 granule vs 无 CER 的 granule

以 20190105 三个 granule 为例：

| 时间 (UTC) | SZA mean | CER(std) 有效 | CER_16 有效 | CTH 有效 | CLP 有效 | Conf Cloudy |
|-----------|----------|:----------:|:----------:|:-------:|:------:|:-----------:|
| 0000 | 8762 | 99,279 (3.6%) | — | 2,748,620 (100%) | 2,748,620 | 2,185,659 (79.5%) |
| 1440 | 12029 | **0 (0%)** | 0 | 2,748,620 (100%) | 2,748,620 | 1,773,842 (64.5%) |
| 1745 | 15386 | **0 (0%)** | 0 | 2,748,620 (100%) | 2,748,620 | 1,255,046 (45.7%) |

注：SZA 值为 MYD06 内部 scaled integer，非度数。

### 有 CER 数据的白天 granule（20191215 0615 UTC）

| CER 变体 | 有效像元 | 覆盖率 |
|----------|---------|:------:|
| `Cloud_Effective_Radius` (2.1µm std) | 1,877,711 | **68.3%** |
| `Cloud_Effective_Radius_37` (3.7µm) | 1,886,052 | **68.6%** |
| `Cloud_Effective_Radius_16` (1.6µm) ← **当前使用** | 1,044,529 | **38.0%** |
| `Cloud_Effective_Radius_1621` (1.6+2.1) | 744,051 | 27.1% |
| `Cloud_Effective_Radius_PCL` | 152,150 | 5.5% |

### 根本原因：CER/COT = 0 是因为夜间 granule

MODIS CER/COT 反演依赖太阳反射通道（1.6µm, 2.1µm, 3.7µm），**夜间完全不可用**。

1月北半球：
- 0000 UTC = 08:00 北京时间 → 日出后，低 CER 覆盖（3.6%）
- 0615 UTC = 14:15 北京时间 → 正午，CER 覆盖好（68.3% std, 38.0% _16）
- 1440 UTC = 22:40 北京时间 → 夜间，无太阳能反射 → CER/COT 全为 0
- 1745 UTC = 01:45 北京时间 → 深夜，无太阳能反射 → CER/COT 全为 0

**结论：管道日志中大量 `cer=0 cot=0` 是客观事实，不是 bug。这些 granule 是夜间的，MODIS 光学反演根本无法产出 CER/COT。**

---

## CLP 监督分析

### CLP_1km 值域（所有 granule 一致）

| 值 | 含义 |
|:--|------|
| 0 | Clear（晴空） |
| 1 | Water（水云） |
| 2 | Ice（冰云） |
| 6 | Undetermined（未定） |

**未出现值 3 (Mixed) 和 4 (Ice_16) 和 5 (Water_16)。**

- `Cloud_Phase_Infrared_1km`：100% 覆盖（无 fill 值出现），fill=127 但从未出现
- `Cloud_Phase_Optical_Properties`：白天 granule 覆盖 99.9%（值 1–4），夜间覆盖接近 0%

### CLP 分类监督充足，但 CER/COT 回归监督只有白天

- 白天 granule：CLP、CTH、CER、COT 全部可用
- 夜间 granule：只有 CLP + CTH 可用，CER/COT 全为 0

**这也是为什么管道中 CLP sample count 看起来 OK 但 CER/COT 回归 R≈0 的原因：**
- 训练数据混杂了大量夜间样本（有 CLP 无 CER/COT）
- 夜间样本 CER/COT 被设为 0 → 模型学到的是"晴空像元 CER/COT=0"和"夜间 cloudy 像元 CER/COT=0"混在一起
- 只有白天样本才提供有效的 CER/COT 回归监督

---

## 当前 config.py 使用建议

```python
MODIS_VARS = {
    "CER": "Cloud_Effective_Radius_16",    # ← 覆盖率 38%（白天）
    "COT": "Cloud_Optical_Thickness_16",   # ← 覆盖率 38%（白天）
}
```

改为标准 2.1µm 可将白天 CER/COT 覆盖率从 38% → 68.3%：

```python
MODIS_VARS = {
    "CER": "Cloud_Effective_Radius",       # ← 覆盖率 68.3%（白天）
    "COT": "Cloud_Optical_Thickness",      # ← 覆盖率 68.3%（白天）
}
```

但这**不解决夜间 granule 没有 CER/COT 的根本问题**——那是物理限制，无法修复。

---

## 实际影响

1. **fuse 阶段**：夜间 granule 的 AGRI 匹配片 → CER/COT 全为 0/NaN → 回归监督缺失
2. **train 阶段**：混合白天/夜间 patch → 回归 loss 只对白天有效 → 模型学到噪声
3. **val OA 45–48%**：CLP 分类尚可但 Water 类极差（27% recall），原因可能是夜间样本的 CLP 在只有 IR 通道时本身就不可靠
4. **CER R=0.10, COT R=0.04**：回归基本失败，需要白天-only 策略或专门处理
