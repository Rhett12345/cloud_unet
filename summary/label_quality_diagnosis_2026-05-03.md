# Label Quality & Fusion Diagnosis — 2026-05-03

## 核心发现

### 1. MODIS CER/COT SDS 选择问题 ✅ 已修复

`config.py` 中 `MODIS_VARS` 的 CER/COT 从 `_16` (1.6μm) 改为 combined (2.1μm)：

| SDS 变体 | Confident Cloudy 覆盖率 | 相关性 vs _16 |
|----------|------------------------|---------------|
| `Cloud_Effective_Radius_16` | 24.6% | — |
| `Cloud_Effective_Radius` (2.1μm) | **56.1%** | r=0.88 |
| `Cloud_Effective_Radius_37` (3.7μm) | 64.1% | r=0.79 |

- Uncertainty SDS `Cloud_Effective_Radius_Uncertainty` (无后缀) 与 combined 产品匹配，换 SDS 后顺带修复了之前 uncertainty 不匹配的 bug
- scale_factor 所有变体一致 (0.01)，valid_range 一致 [0, 10000]

### 2. Combined CER 多出来的像素质量正常

| 质量指标 | 与 CER_16 重叠的像素 | combined 独有的像素 |
|----------|---------------------|---------------------|
| CER p50 | 11.1 μm | 13.7 μm |
| UNC p50 | 6.5% | 6.8% |
| UNC > 30% 比例 | 2.3% | 2.8% |

结论：**combined CER 额外覆盖的像素质量与 CER_16 没有区别，不是噪声源。**

### 3. 但换 SDS 后模型效果下降了

训练结果对比（同样的模型、同样的数据量）：

| 指标 | CER_16 (旧) | combined (新) |
|------|------------|---------------|
| CLP OA | 48.6% | 45.0% |
| CLP macro | 47.5% | 44.4% |
| CER R | 0.247 | 0.169 |
| COT R | 0.146 | 0.131 |
| CTH R | 0.275 | 0.177 |
| CER_n (test) | 1.4M | 2.6M |

CER_n 增加了 86% 但所有指标全面下降。

### 4. 根因分析

**AGRI BT 与 CER 的物理相关性弱：**

| 相态 | BT(10.8μm) vs CER r |
|------|---------------------|
| Water (单独) | 0.34 |
| Ice (单独) | 0.33 |
| 不区分相态 | ~0 |

CLP 分类准确率仅 45%，模型经常用错误的相态模式预测 CER → 级联错误。

**CER_16 只对"容易"的像素成功检索**（BT-CER 相关性强的），combined CER 覆盖了更多"难"像素 → 拉低整体 R。

### 5. 模型容量验证

- 1.97M 参数的 U-Net 在随机数据上可达到 98% CLP 准确率（过拟合测试通过）
- 说明模型容量足够拟合，问题在于**真实标签信号太弱/噪声太大**

---

## 融合端核心问题：k=1 单像元噪声

### 当前做法

`fusion_core.py:432`：每个 AGRI 4km 像元在 2.5km 半径内只取 **1 个最近的** MODIS 1km 像元：

```python
tree.query(a_xyz, k=1, distance_upper_bound=chord)
```

### 问题

- 一个 AGRI 4km 像元 ≈ 16 个 MODIS 1km 像元面积
- 2.5km 搜索半径内平均有 **~10 个** MODIS 邻居
- 取最近的那 1 个 → 23% 的像元 CER 误差 > 5μm，7.6% 误差 > 10μm

### 建议修改：k=5 中位数/众数

改两处（只改 `fusion_core.py`）：

**位置 1 — `_collect_1km()` line 432：**

```python
# k=1 → k=5
dist_chord, nn_idx = tree.query(a_xyz, k=5, distance_upper_bound=chord, workers=1)
```

**位置 2 — `aggregate_modis_to_agri()` line 306-313：**

```python
# CLP → 众数 (scipy.stats.mode)
# CER/COT/CTH → np.nanmedian
best_clp = stats.mode(clp_v, keepdims=False).mode
best_cer = float(np.nanmedian(cer_v))
best_cot = float(np.nanmedian(cot_v))
best_cth = float(np.nanmedian(cth_v))
```

### 代价

- 92% 的 AGRI 像元已有 ≥5 个 MODIS 邻居（无需扩大搜索半径）
- 只改 `fusion_core.py`，下游 `fusion_io.py` / `data_fusion.py` / `dataset.py` / `train.py` 完全不受影响
- 输出 dict 结构不变

---

## 其他观察

### 融合成功率低

- 某天 26 个任务只有 5 个成功产出 H5（19%）
- 205 个场景 post-qc `clp=0`（`dt_mean=nanmin`），全部计算浪费
- 原因：MODIS granule 与 AGRI 圆盘时间/空间不匹配

### 训练状态

- 18-21 epoch early stop，val macro_acc 最高仅 35%（随机基线 33%）
- CLP Water recall 39%，Ice recall 39%
- CER/COT/CTH R 值均 < 0.2

---

## 待决策

1. **是否立即实施 k=5 中位数**？（预计减少 23% 的 >5μm 标签噪声）
2. **CER SDS 是否回退到 `_16`**？（牺牲覆盖率换模型预测精度）
3. **是否需要增大模型**？（base_ch 16→32，参数量 1.97M → ~7.8M）
4. **是否加入 uncertainty-weighted loss**？（利用 MODIS 自带 uncertainty 降权）
