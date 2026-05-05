# QC 简化 + 地理可视化改造 — 2026-05-04

## 背景

参照 GeoISCLD-Net 原始代码，发现其训练用的云标签来自 AGRI L2 产品（同仪器、同时刻、同网格），而本管线的标签来自 MODIS MYD06（跨仪器、时间差、空间重采样），这是两个管道在标签来源上的根本差异。

GeoISCLD-Net 的 QC 链非常简单：仅 SZA<65° + VZA<65° + CLP 有效范围 + CER/COT 非填充值 + 晴空回归=0。没有 Cloud_Mask 比特解码、光学相态一致性、CTH 辅助过滤等复杂步骤。

## 已完成的改动

### 1. 简化 QC 链（config.py）

三处开关翻转：

```python
MODIS_FILTER_WEAK_QUALITY = False     # 曾是 True — 关闭弱质量过滤
MODIS_REQUIRE_OPTICAL_PHASE_FOR_COP = False  # 曾是 True — 关闭光学相态过滤
MODIS_REQUIRE_CTH_AUX = False         # 曾是 True — 关闭 CTH 辅助过滤
```

这三个开关在 `fusion_io.py` 的 `_apply_qa_filter()` 中 gate 对应过滤块，翻 False 后过滤块被跳过。

### 2. 统一时间窗口（fusion_config.py）

```python
REG_TIME_MAX_MIN = 7.5  # 曾为 5.0 — 与 CLP 分类的 TIME_LOW_Q_MIN 统一
```

### 3. 晴空回归监督 = 0（fusion_io.py）

```python
# 参照 GeoISCLD-Net：晴空像元 CER/COT/CTH 设为 0（而非 NaN）
# 让模型学到 "晴空 = 没有云光学/几何属性"
for k in ["CER", "COT", "CTH"]:
    labels[k][~cloudy] = 0.0
```

### 4. 地理可视化改为单图验证（data_fusion.py）

重写了 `_make_geo_figure`，从原来的双面板（geo + QC bar chart）改为单面板地理叠加图。

**图上逐层元素：**

| 图层 | 代码位置 | 内容 |
|------|---------|------|
| 浅灰散点 | `_make_geo_figure` L337-339 | AGRI 全圆盘经纬度网格，降采样到 ~4000 点，s=0.25 |
| 深蓝轮廓 | `_draw_disk_outline` L459-484 | AGRI 有效区域外轮廓，72 角度分箱 concave hull |
| 彩色填充+虚线 | `_make_geo_figure` L346-368 | MODIS 条带轮廓 + [IN]/[OUT] 判定 |
| 深蓝方块+坐标 | `_annotate_disk_edges` L422-456 | N/E/S/W 四边各 3 个采样点，标注经纬度 |
| MODIS 四角标注 | `_make_geo_figure` L370-378 | 每个 granule bounding box 四角经纬度 |

**新增辅助函数：**
- `_compute_swath_outline()` L253 — MODIS 有效区域 concave hull
- `_check_modis_border_in_agri()` L279 — 检查 MODIS 边缘是否落在 AGRI 圆盘内
- `_annotate_disk_edges()` L422 — AGRI 圆盘 N/E/S/W 边界采样标注
- `_draw_disk_outline()` L459 — AGRI 圆盘外轮廓

**移除：** `_make_qc_figure`（约 90 行）

### 5. 其他代码+（fusion_io.py）

- 新增 `read_modis_geo_quick()` — 轻量地理预检，只读经纬度
- `read_myd06()` 新增 `geo_cache` 参数 — 避免重复读取经纬度
- Uncertainty SDS 名称修正 — `_16` 产品用 `_Uncertainty_16`，非 `_16` 用 `_Uncertainty`

---

## 当前可视化的问题

### 图上的"蓝色点"

暗蓝色方块来自 `_annotate_disk_edges`（N/E/S/W 各 3 个采样点，共 12 个），外加 `_draw_disk_outline` 的 royalblue 轮廓线。从像素分析来看图上检测到约 53 处蓝色像素聚类。

**算法问题：** `_annotate_disk_edges` 用 `arctan2(y - center_lat, x - center_lon)` 做角度分箱找 N/E/S/W 边缘，12 个点过于密集，且边缘数据缺失时选出的点可能不在真正的边界上，导致位置看起来诡异。

### "拳头形状"

这是**几何上正确的**：AGRI 全圆盘是一个大圆（地球圆盘），MODIS 极轨条带是一个弧形长条。1-2 个 MODIS granule 部分重叠在 AGRI 圆盘上 → 圆形+弧形=拳头形状。不是 bug。

### 图幅问题

图像仅 1427×689 像素，内容（非白像素）只占 9.8%。tight_layout 裁切后有效绘图区域偏小。

---

## 待完成

1. **用简化 QC 重跑融合管道** — `fuse → stats → train`，看 val OA 能否突破 45%
2. **修复 20190125 的损坏 H5**（6 通道旧格式导致训练崩溃）
3. **简化地理图** — 用户反馈蓝色点太多太乱，建议只保留 AGRI 轮廓 + MODIS 条带 + 四角经纬度 + IN/OUT 结论

---

## 关键环境变量

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `FUSION_REG_TIME_MAX_MIN` | 7.5 | 回归标签时间窗口 (min) |
| `FUSION_AGRI_SEARCH_RADIUS_KM` | 2.5 | MODIS→AGRI 匹配半径 (km) |
| `FUSION_DISK_MARGIN_DEG` | 5.0 | AGRI 圆盘边缘收缩 (deg) |
| `ENABLE_QC_DIAGNOSTICS` | false | 开启 QC 诊断 CSV/JSONL |

---

## 已知未解决

- 修改前 `data_fusion.py` 的 type hints 和 docstring 省略了部分新参数的说明
- main.py 仍接受 `--max_qc` 参数但已无效（仅向后兼容）
