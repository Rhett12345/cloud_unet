"""
AGRI L2 CLP vs MODIS 验证可视化 — 双方案合并为一张 2×2 图。

用法:
    python plot_validation_figures.py
输出:
    validation_report.{svg,pdf,tiff,png}
"""

import sys
from pathlib import Path
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════
# Nature 风格设置
# ═══════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# ═══════════════════════════════════════════════════════════════════════
# 统一字号：3 档
# ═══════════════════════════════════════════════════════════════════════
FS_MAIN  = 7    # 轴标签、tick、bar 数值
FS_ANNOT = 6    # 次要标注
FS_SMALL = 5.5  # colorbar 等最小级

# ═══════════════════════════════════════════════════════════════════════
# 色板
# ═══════════════════════════════════════════════════════════════════════
C_BLUE    = "#0F4D92"
C_GREEN   = "#2E9E44"
C_RED     = "#E53935"
C_TEAL    = "#42949E"
C_ORANGE  = "#E8871D"
C_NEUTRAL = "#767676"
C_LIGHT   = "#CFCECE"

# ═══════════════════════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════════════════════
ALL_SCENES = [
    {"ts": "03:00", "l2_cf": 69.2, "cm_cf": 92.2, "oa": 88.1, "n": 203862},
    {"ts": "06:00", "l2_cf": 71.1, "cm_cf": 90.1, "oa": 86.7, "n":  97755},
    {"ts": "06:15", "l2_cf": 70.9, "cm_cf": 47.2, "oa": 47.1, "n": 446605},
    {"ts": "08:00", "l2_cf": 69.0, "cm_cf": 63.2, "oa": 50.3, "n": 364524},
    {"ts": "15:15", "l2_cf": 64.1, "cm_cf": 70.6, "oa": 64.6, "n": 156425},
    {"ts": "17:00", "l2_cf": 83.3, "cm_cf": 68.5, "oa": 56.3, "n": 393832},
    {"ts": "22:00", "l2_cf": 65.5, "cm_cf": 46.2, "oa": 41.2, "n": 209806},
]

# 场景筛选: "all" | "cm_gt_l2" (MODIS 云量 > L2 云量, 效果好) | 自定义 ts 列表 eg. ["03:00","06:00"]
SCENE_FILTER = "all"

def _apply_filter(scenes, rule):
    if rule == "all":
        return scenes
    if rule == "cm_gt_l2":
        return [s for s in scenes if s["cm_cf"] > s["l2_cf"]]
    if isinstance(rule, (list, tuple)):
        names = set(rule)
        return [s for s in scenes if s["ts"] in names]
    raise ValueError(f"Unknown SCENE_FILTER: {rule}")

SCENES = _apply_filter(ALL_SCENES, SCENE_FILTER)

# Pooled 混淆矩阵 (全场景)
CM_ALL = np.array([[543008, 538363],
                    [210412, 136200]])

# 如果用筛选，提示 CM 也需对应更新
if SCENE_FILTER != "all":
    print(f"SCENE_FILTER={SCENE_FILTER!r}, using {len(SCENES)}/{len(ALL_SCENES)} scenes: "
          f"{[s['ts'] for s in SCENES]}")
    print("WARNING: CM still uses all-scene pooled values. "
          "Update CM manually if per-scene confusion matrices are available.")

CM = CM_ALL

tp, fp = CM[0, 0], CM[0, 1]
fn, tn = CM[1, 0], CM[1, 1]
CDR = tp / (tp + fn) * 100
FAR = fp / (fp + tn) * 100
OA  = (tp + tn) / CM.sum() * 100

HIGH_CF = [s for s in SCENES if s["cm_cf"] > 60]
LOW_CF  = [s for s in SCENES if s["cm_cf"] <= 60]

def _pooled_oa(ss):
    total_n = sum(s["n"] for s in ss)
    if total_n == 0:
        return 0.0
    return sum(int(s["n"] * s["oa"] / 100) for s in ss) / total_n * 100

OA_HIGH = _pooled_oa(HIGH_CF)
OA_LOW  = _pooled_oa(LOW_CF)


# ═══════════════════════════════════════════════════════════════════════
# 主图
# ═══════════════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(170 / 25.4, 130 / 25.4))
    gs = GridSpec(2, 2, wspace=0.38, hspace=0.48,
                  left=0.09, right=0.97, top=0.93, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_cdr_far(ax_a)
    _label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_cm(ax_b)
    _label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    _draw_stratified_oa(ax_c)
    _label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 1])
    _draw_cf_scatter(ax_d)
    _label(ax_d, "d")

    out = Path(__file__).resolve().parent / "validation_report"
    for fmt, kw in [("svg", {}), ("pdf", {}),
                    # ("tiff", {"dpi": 600}),
                    ("png", {"dpi": 300})]:
        fig.savefig(f"{out}.{fmt}", bbox_inches="tight", **kw)
    print(f"Saved: {out}.{{svg,pdf,png}}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 子图
# ═══════════════════════════════════════════════════════════════════════
def _label(ax, s):
    ax.set_title(s, fontweight="bold", fontsize=8, loc="left", pad=4)


def _draw_cdr_far(ax):
    """
    a: CDR & FAR 水平条形图
    修改：
    - 删除右侧浮动描述文字，改用 y 轴副标签传达语义
    - x 轴还原 [0, 110]，消除比例失真
    - 删除 axvline(50) 参考线
    - 统一字号
    """
    labels  = ["CDR", "FAR"]
    vals    = [CDR, FAR]
    colors  = [C_GREEN, C_RED]
    # sublbls = ["MODIS Cloud → L2 Cloud", "MODIS Clear → L2 Cloud"]

    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.42, edgecolor="none")
    ax.barh(y, [100 - v for v in vals], left=vals,
            color=[C_LIGHT] * 2, height=0.42, edgecolor="none", alpha=0.45)

    for bar, val in zip(bars, vals):
        ax.text(val - 2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="right", va="center",
                fontsize=FS_MAIN, fontweight="bold", color="white")

    ax.set_xlim(0, 110)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_MAIN, fontweight="bold")
    ax.set_xlabel("Rate (%)", fontsize=FS_MAIN)
    ax.invert_yaxis()

    # 用 y 轴副标签替代浮动文字
    # ax2 = ax.secondary_yaxis("right")
    # ax2.set_yticks(y)
    # ax2.set_yticklabels(
    #     sublbls,
    #     fontsize=FS_ANNOT, color=C_NEUTRAL,
    #                     style="italic")
    # ax2.tick_params(length=0)
    # for spine in ax2.spines.values():
    #     spine.set_visible(False)


def _draw_cm(ax):
    """
    b: 混淆矩阵热力图
    修改：
    - 删除脱出轴范围的角标文字（TP/FP/FN/TN）
    - 改为在单元格内叠加小字说明，颜色与热力图协调
    - tick 字号统一
    """
    cm_pct = CM / CM.sum() * 100
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=55, aspect="auto")

    cell_labels = [["Hit (TP)", "False Alarm (FP)"],
                   ["Miss (FN)", "Correct Rej. (TN)"]]
    cell_colors = [[C_GREEN, C_RED],
                   [C_ORANGE, C_TEAL]]

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_pct[i, j] > 30 else "black"
            # 主数值
            ax.text(j, i - 0.12, f"{CM[i, j]:,}",
                    ha="center", va="center",
                    fontsize=FS_MAIN, color=text_color, fontweight="bold")
            # 百分比
            ax.text(j, i + 0.18, f"({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=FS_ANNOT, color=text_color)
            # 单元格类型标签（替代原角标）
            lbl_color = "white" if cm_pct[i, j] > 30 else cell_colors[i][j]
            ax.text(j, i + 0.42, cell_labels[i][j],
                    ha="center", va="center",
                    fontsize=FS_SMALL, color=lbl_color, style="italic")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MODIS Cloud", "MODIS Clear"], fontsize=FS_ANNOT)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["L2 Cloud", "L2 Clear"], fontsize=FS_ANNOT)
    ax.tick_params(length=0)


def _draw_stratified_oa(ax):
    """
    c: 分层 Pooled OA 柱状图
    修改：
    - 删除双向箭头与 Δ bbox，改用简洁文字 caption
    - 删除 axhline(50) 参考线
    - 统一字号
    """
    strata = [
        ("Overall", OA, C_BLUE),
        ("MODIS CF\n> 60%", OA_HIGH, C_TEAL),
        ("MODIS CF\n≤ 60%", OA_LOW, C_ORANGE),
    ]
    x      = np.arange(len(strata))
    vals   = [v for _, v, _ in strata]
    colors = [c for _, _, c in strata]
    labels = [l for l, _, _ in strata]

    bars = ax.bar(x, vals, color=colors, width=0.48, edgecolor="none")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,
                f"{val:.1f}%", ha="center", fontsize=FS_MAIN, fontweight="bold")

    # 场景数标注
    # ax.text(1, OA_HIGH + 6.5,
    #         # f"n={len(HIGH_CF)}",
    #         ha="center", fontsize=FS_ANNOT, color=C_TEAL)
    # ax.text(2, OA_LOW + 6.5,
    #         # f"n={len(LOW_CF)}",
    #         ha="center", fontsize=FS_ANNOT, color=C_ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_ANNOT)
    ax.set_ylabel("Pooled OA (%)", fontsize=FS_MAIN)
    ax.set_ylim(0, 108)

    # 用 ax.text caption 替代箭头+bbox
    # delta = OA_HIGH - OA_LOW
    # ax.text(0.97, 0.04,
    #         f"Δ(high – low CF) = {delta:.1f} pp",
    #         transform=ax.transAxes, ha="right", va="bottom",
    #         fontsize=FS_ANNOT, color=C_NEUTRAL, style="italic")


def _draw_cf_scatter(ax):
    """
    d: L2 vs MODIS 云量散点图
    修改：
    - 删除星号 annotate（改用描边区分 high/low CF）
    - 删除斜向浮动文字 "L2 over/under-estimates"
    - colorbar 用 fraction/pad 替代 shrink，比例更协调
    - 统一字号
    """
    l2_cf = [s["l2_cf"] for s in SCENES]
    cm_cf = [s["cm_cf"] for s in SCENES]
    oa_v  = [s["oa"] for s in SCENES]
    sizes = [max(30, s["n"] / 8000) for s in SCENES]

    # high/low CF 用描边粗细区分
    edge_lw = [1.2 if s["cm_cf"] > 60 else 0.4 for s in SCENES]
    edge_c  = [C_TEAL if s["cm_cf"] > 60 else "white" for s in SCENES]

    # y=x 参考线
    ax.plot([35, 100], [35, 100], ls="--", color=C_LIGHT, lw=0.7, zorder=0)

    sc = ax.scatter(l2_cf, cm_cf, c=oa_v, cmap="RdYlGn", vmin=35, vmax=95,
                    s=sizes, edgecolors=edge_c, linewidths=edge_lw, zorder=2)

    # for s in SCENES:
    #     ax.annotate(s["ts"], (s["l2_cf"], s["cm_cf"]),
    #                 textcoords="offset points", xytext=(5, 3),
    #                 fontsize=FS_ANNOT, color=C_NEUTRAL)

    ax.set_xlabel("AGRI L2 Cloud Fraction (%)", fontsize=FS_MAIN)
    ax.set_ylabel("MODIS CM Cloud Fraction (%)", fontsize=FS_MAIN)
    ax.set_xlim(40, 96)
    ax.set_ylim(40, 96)
    ax.set_aspect("equal")

    # colorbar：fraction/pad 方案，避免 shrink 导致比例不协调
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("OA (%)", fontsize=FS_ANNOT)
    cb.ax.tick_params(labelsize=FS_SMALL)

    # 图例：描边说明 high/low CF
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker="o", color="none",
    #            markerfacecolor=C_NEUTRAL, markeredgecolor=C_TEAL,
    #            markeredgewidth=1.2, markersize=5,
    #            label="MODIS CF > 60%"),
    #     Line2D([0], [0], marker="o", color="none",
    #            markerfacecolor=C_NEUTRAL, markeredgecolor="white",
    #            markeredgewidth=0.4, markersize=5,
    #            label="MODIS CF ≤ 60%"),
    # ]
    # ax.legend(handles=legend_elements, fontsize=FS_ANNOT,
    #           loc="lower right", handletextpad=0.5, borderpad=0.6)


if __name__ == "__main__":
    main()
