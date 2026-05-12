# MODIS 5km CTH 外部验证 — 问题与方向

**日期**: 2026-05-12  
**状态**: 验证代码逻辑已正确，匹配策略已优化（±5min + 圆盘内过滤），但模型 vs MODIS CTH 相关性接近零的根本原因待确认

---

## 1. 背景

用原始 GeoISCLD-Net 的 `MODISCOMPmatched.py` 方法论重写了 `validate_modis.py` 和 `validate_modis_cloud.py`：

- 使用 MODIS MYD06 **5km** `Cloud_Top_Height`（非 1km）
- KD-tree 最近邻匹配（4km 搜索半径，(lat, lon) 度空间，k=1）
- 只验证 CTH，不做云检测验证
- 所有函数 inline，不依赖 `fusion_io` / `fusion_core`

## 2. 当前结果

### 模型性能（对 AGRI L2 CTH 训练标签）

| 指标 | 值 |
|------|-----|
| CTH R | **0.9598** |
| CTH RMSE | 1191 m |
| CTH Bias | +30.5 m |
| CLP OA | 86.3% |

→ 模型训练充分，对训练标签的预测能力很好。

### 模型 vs MODIS 5km CTH（±15min, 无圆盘过滤）

| 指标 | 值 |
|------|-----|
| CTH R | **-0.014 ± 0.256** |
| CTH RMSE | 6185 ± 1911 m |
| CTH Bias | +1418 ± 4069 m |
| 总匹配像素 | 3,524,630 (36 scenes) |

→ 完全没有相关性，且 scene 间方差极大。

### 单场景深挖（20190505_000000）

```
AGRI Earth disk 有效像素:  5,761,592
成功匹配 MODIS 的像素:         7,647  (0.1%)
全部位于纬度 40-80°N
```

| MODIS CTH bin | 模型预测均值 | MODIS 均值 | 趋势 |
|---------------|:---------:|:--------:|:----:|
| 0–1000 m | 10433 m | 394 m | **反相关** |
| 1000–3000 m | 8730 m | 2044 m | |
| 3000–5000 m | 7786 m | 3907 m | |
| 8000–12000 m | 7612 m | 8972 m | |
| 12000–20000 m | 7405 m | 13366 m | **反相关** |

| 纬度带 | 模型均值 | MODIS 均值 | 趋势 |
|:------:|:-------:|:---------:|:----:|
| 40–60°N | 6737 m | 4611 m | |
| 60–80°N | 10068 m | 2776 m | 纬度↑ 模型↑ MODIS↓ |

## 3. 已排除的问题

- [x] **CTH 单位/scale**: MODIS `Cloud_Top_Height` scale_factor=1.0, valid_range=[0,18000], units=meters — 无需缩放
- [x] **空间匹配错误**: npz 经纬度 = GEO 文件经纬度（diff=0.0），KD-tree 实现与原始代码一致
- [x] **GEO 读取失败**: 已添加 `_derive_latlon` 回退，36/36 scenes 成功
- [x] **MODIS CTH=0 是 fill value**: `_FillValue=-32767`，0 属于 valid_range 内的合法低值
- [x] **验证代码逻辑**: 与原始 `MODISCOMPmatched.py` 的 KD-tree / 5km / CTH-only 方法一致

## 4. 已做的优化（待测试）

- [ ] **时间窗口**: 15min → **5min**（±5min 内最多 3 个 MODIS granule）
- [ ] **圆盘内过滤**: 只保留 ≥95% 像素落入 AGRI 圆盘（角距 ≤75°）的 MODIS granule
- [ ] **使用 npz 自带经纬度**: 不再从 GEO 文件重新计算

预期效果：排除圆盘边缘（大 VZA）和 MODIS swath 边缘（大扫描角）的双重低质量区域。

## 5. 待解答的核心疑问

### 5.1 AGRI L2 CTH 和 MODIS 5km CTH 到底有没有可比性？

**最关键的悬而未决的问题。** 需要运行：
```bash
python validate_modis_cloud.py --day 20190505 --reference l2 --summary
```
如果 L2 直接 vs MODIS 也是 R≈0，说明两个产品在这个区域确实不相关。

### 5.2 为什么 MODIS CTH 低值区模型预测高云？

MODIS CTH=0~1000m 时模型预测 ~10433m。可能原因：
- MODIS CTH=0 是反演失败的默认值（非真值）
- AGRI L2 CTH 在高纬度/大 VZA 时系统性偏高
- 两个产品对薄卷云/多层云的响应完全不同

### 5.3 原始 GeoISCLD-Net 的 CTH_true 标签来源是什么？

如果原始用 MODIS 自身作为训练标签（CTH_true = MODIS CTH），那验证时的 R 自然高。我们的模型学的是 AGRI L2 CTH，和 MODIS 没有直接关系。

### 5.4 匹配覆盖率 0.1% 的问题是否由 MODIS 轨道决定？

MODIS（Aqua）过境时间约 13:30 LT。2019-05-05 00:00 UTC 对应东亚约 08:00 LT。MODIS swath 覆盖的经度带是否与 AGRI 圆盘中心 (104.7°E) 匹配？需要检查 MODIS 过境轨道与 AGRI 视场的几何重叠。

### 5.5 MODIS CTH=0 是否需要特殊处理？

MODIS `Cloud_Top_Height` long_name 说 "rounded to nearest 50m"。CTH=0 在地理意义上不存在（云顶不可能在地表）。可能需要 `CTH > 0` 的额外过滤。

## 6. 可能的后续方向

### 6.1 直接对比 AGRI L2 CTH vs MODIS CTH（最高优先级）

```bash
python validate_modis_cloud.py --day 20190505 --reference l2 --summary
python validate_modis_cloud.py --day 20190505 --reference model --npz_dir /path/ --summary
```

如果 L2 和模型对 MODIS 的验证结果相似，说明问题在数据产品层面，非模型层面。

### 6.2 扩大时间窗口做敏感性测试

试试 ±15min, ±30min, ±60min，看结果是否对时间窗口敏感。

### 6.3 滤除 MODIS CTH=0

在匹配前 `cth[(cth <= 0) | (cth > 20000)] = np.nan` → `cth[(cth < 1) | (cth > 20000)] = np.nan`，排除可能的反演失败值。

### 6.4 仅用低纬度区域验证

只匹配 |lat| < 60° 的像素，避开 AGRI 圆盘边缘和 MODIS 高纬度扫描角。

### 6.5 对比 AGRI L2 CTH 和 MODIS CTH 的散点图

直接画出 L2 CTH vs MODIS CTH 的密度散点，定性诊断两者关系的模式（线性、非线性、还是纯噪声）。

### 6.6 使用 CER/COT 联合过滤

参考原始 GeoISCLD-Net：要求 CER、COT、CTH 三者同时有效，减少单产品反演失败的污染。

---

## 附录：修改过的文件

| 文件 | 主要改动 |
|------|---------|
| `validate_modis.py` | TIME_WINDOW=5min, AGRI disk gate, npz lat/lon, _derive_latlon 回退 |
| `validate_modis_cloud.py` | 同上 + model/l2 双模式 |
