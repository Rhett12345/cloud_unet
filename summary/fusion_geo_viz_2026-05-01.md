# Fusion Geo Visualization — 融合地理定位验证图

**日期**: 2026-05-01
**分支**: master
**改动文件**: `data_fusion.py`

---

## 目标

在数据融合（MODIS → AGRI）完成后，自动生成一张带经纬度标注的地理定位验证图，用于快速排查 MODIS 数据是否正确、完整地落入 AGRI 全圆盘范围内。作为融合主流程的补充，不影响原有逻辑。

---

## 改动详情

### 修改文件：`data_fusion.py`

#### 1. 新增 `_draw_disk_outline(ax, lat, lon, **kwargs)`（第 311–336 行）

在经纬度坐标轴上画出 AGRI 有效像元的外轮廓线。

- 方法：极角分箱近似凸包。以有效像元中心为原点，将 360° 等分为 72 个扇形区间，每个区间取距中心最远的点，按角度排序连线。
- 用于在 lat/lon 地图上标出 AGRI 全圆盘的物理边界。

#### 2. 新增 `_make_geo_figure(agri, labels, agri_dt, modis_bounds, save_path)`（第 218–308 行）

生成一张 1×2 的地理验证图，保存为 PNG。

**左图 — 经纬度空间覆盖图**：
- 灰色散点：AGRI 有效像元（下采样至 ~6000 点，作为圆盘背景参考）
- 蓝色实线：AGRI 圆盘外轮廓（`_draw_disk_outline`）
- 彩色散点：MODIS CLP 覆盖（按相态着色：灰=Clear, 蓝=Water, 橙=Ice，下采样至 ~8000 点）
- 虚线矩形：MODIS 原始条带的 lat/lon 外包边界框（每个 MODIS granule 一个）
- 经纬度网格线
- `ax.set_aspect("equal")` 保持经纬度比例

**右图 — 覆盖统计**：
- 相态分布柱状图（Clear / Water / Ice 像素计数）
- 文字信息：AGRI 时间、有效像元数、MODIS 覆盖率、各相态像素、每个 MODIS 条带经纬度范围

**异常安全**：整个函数包裹在 try/except 中，matplotlib 错误只记 warning 日志，绝不抛出异常影响融合主流程。

#### 3. `_fuse_one_scene()` 两处微调

**第 157–171 行** — MODIS 条带边界框收集：
在 MODIS 读取循环完成后、调用 `aggregate_modis_to_agri()` 之前，遍历 `modis_list`，从每个 MODIS granule 的 1km（或 5km fallback）经纬度中提取有效像元的 lat/lon min/max 范围，存入 `modis_bounds` 列表。这一步必须在聚合前完成，因为聚合后不再保留 MODIS 原始坐标。

**第 205–211 行** — 调用地理可视化：
在成功写出 H5 文件后（无论 `samples_only` 还是 `full_disk` 模式），调用 `_make_geo_figure()`，输出 `*_geo.png` 与 H5 同目录。

---

## 输出物

每个成功融合的场景多出一个文件：

```
<out_dir>/AGRI_MYD06_YYYYMMDD_HHMMSS_geo.png
```

与 QC 诊断图（`*_qc.png`）并列存放。

---

## 当前局限 & 待改进

1. **lat/lon 散点图不够直观**：AGRI 像元在经纬度空间不是均匀网格，散点下采样后看不出标签覆盖的连续性和空洞。
2. **MODIS 条带框是矩形外包**：仅取 lat/lon 的 min/max 画矩形，不能反映 MODIS 条带的真实 swath 形状（倾斜扫描带），可能产生误导。
3. **右图信息密度低**：统计面板占一半画布，但信息用几行文字就能说清。
4. **缺少像素坐标视图**：无法直观看到 MODIS 标签在 AGRI 影像 row/col 像素位置上的分布，不便于定位到具体像元。

### 可能的改进方向

- 改为 **AGRI 像素坐标（row/col）** 视图，类似 `tools/visualize_h5_paired_batch.py` 的思路，同时叠加经纬度网格/刻度线作为参考
- 或将 MODIS 覆盖以 **imshow 方式叠加在 AGRI BT 底图** 上，mask 掉无标签区域
- 去掉右图统计面板，改为单张大地图 + 图上方文字标注关键数字

---

## 与现有工具的关系

| 工具 | 输入 | 输出 | 坐标空间 |
|------|------|------|----------|
| `data_fusion._make_geo_figure`（新增） | 内存中的 agri + labels | `*_geo.png` | lat/lon |
| `data_fusion._make_qc_figure`（已有） | 已写出的 H5 文件 | `*_qc.png` | row/col |
| `tools/visualize_fusion_geo.py` | 已写出的 H5 文件 | `*_geo.png` | lat/lon |
| `tools/visualize_h5_paired_batch.py` | 已写出的 H5 文件 | `*_modis_region.png` | row/col |

新增的 `_make_geo_figure` 与 `tools/visualize_fusion_geo.py` 功能相似（都是 lat/lon 空间），区别在于：
- 集成在融合流程内部，使用内存数据，无需事后单独运行
- 额外画了 MODIS 条带边界框
- 融合完成即出图，适合批量运行时快速扫一眼定位问题场景
