"""
AGRI L2 CLP vs MODIS 验证可视化 — 精致重绘版
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# ═══════════════════════════════════════════════════════════════════════
# 全局样式
# ═══════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#AAAAAA",
    "legend.frameon": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.labelcolor": "#333333",
    "text.color": "#222222",
})

FS_MAIN  = 7
FS_ANNOT = 6
FS_SMALL = 5.5

# ═══════════════════════════════════════════════════════════════════════
# 配色方案
# ═══════════════════════════════════════════════════════════════════════
# 主色调：深海蓝系 + 珊瑚红点缀
C_NAVY    = "#1A3A5C"
C_BLUE    = "#2166AC"
C_SKYBLUE = "#74B3CE"
C_TEAL    = "#1B9E8A"
C_GREEN   = "#2A9D60"
C_LIME    = "#8BC34A"
C_RED     = "#D62728"
C_CORAL   = "#E07050"
C_AMBER   = "#E8A020"
C_PURPLE  = "#7B5EA7"
C_NEUTRAL = "#888888"
C_LIGHT   = "#E8EDF2"
C_BG      = "#F7F9FC"  # 极浅蓝灰背景

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
SCENES = ALL_SCENES

CM = np.array([[543008, 538363],
               [210412, 136200]])

tp, fp = CM[0, 0], CM[0, 1]
fn, tn = CM[1, 0], CM[1, 1]
CDR = tp / (tp + fn) * 100
FAR = fp / (fp + tn) * 100
OA  = (tp + tn) / CM.sum() * 100

HIGH_CF = [s for s in SCENES if s["cm_cf"] > 60]
LOW_CF  = [s for s in SCENES if s["cm_cf"] <= 60]

def _pooled_oa(ss):
    total_n = sum(s["n"] for s in ss)
    if total_n == 0: return 0.0
    return sum(int(s["n"] * s["oa"] / 100) for s in ss) / total_n * 100

OA_HIGH = _pooled_oa(HIGH_CF)
OA_LOW  = _pooled_oa(LOW_CF)


# ═══════════════════════════════════════════════════════════════════════
# 辅助：渐变矩形填充（模拟渐变条）
# ═══════════════════════════════════════════════════════════════════════
def _gradient_hbar(ax, y, width, height, color_start, color_end, alpha=1.0):
    """用多个小矩形模拟水平渐变条"""
    n_steps = 200
    x_vals = np.linspace(0, width, n_steps + 1)
    r1, g1, b1 = mpl.colors.to_rgb(color_start)
    r2, g2, b2 = mpl.colors.to_rgb(color_end)
    for k in range(n_steps):
        t = k / n_steps
        c = (r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1))
        rect = mpatches.Rectangle((x_vals[k], y - height/2),
                                   x_vals[k+1] - x_vals[k], height,
                                   color=c, alpha=alpha, ec="none")
        ax.add_patch(rect)


# ═══════════════════════════════════════════════════════════════════════
# 主图
# ═══════════════════════════════════════════════════════════════════════
def main():
    fig = plt.figure(figsize=(175/25.4, 135/25.4), facecolor="white")
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 2, wspace=0.35, hspace=0.42,
                  left=0.12, right=0.93, top=0.91, bottom=0.11)

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
    for fmt, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(f"{out}.{fmt}", bbox_inches="tight", **kw)
    print(f"Saved: {out}.{{svg,pdf,png}}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 子图
# ═══════════════════════════════════════════════════════════════════════
def _label(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="top",
            color=C_NAVY,
            # bbox=dict(boxstyle="round,pad=0.15", facecolor=C_LIGHT,
            #           edgecolor="none", alpha=0.8)
            )


def _draw_cdr_far(ax):
    """a: CDR & FAR — 渐变条 + 精致标注"""
    ax.set_facecolor("white")

    labels = ["CDR", "FAR"]
    vals   = [CDR, FAR]
    c_main = [C_TEAL, C_CORAL]
    c_fade = ["#A8D8D0", "#F2C4B4"]

    y = np.array([0.72, 0.28])
    bar_h = 0.22

    for i, (lbl, val, cm, cf) in enumerate(zip(labels, vals, c_main, c_fade)):
        # 背景轨道
        bg = mpatches.FancyBboxPatch((0, y[i]-bar_h/2), 100, bar_h,
                                     boxstyle="round,pad=0.005",
                                     facecolor="#EEF1F5", edgecolor="none")
        ax.add_patch(bg)
        # 渐变填充条
        _gradient_hbar(ax, y[i], val, bar_h, cm, cf, alpha=0.95)
        # 描边
        bdr = mpatches.FancyBboxPatch((0, y[i]-bar_h/2), val, bar_h,
                                      boxstyle="round,pad=0.005",
                                      facecolor="none",
                                      edgecolor=cm, linewidth=0.5, alpha=0.6)
        ax.add_patch(bdr)

        # 数值标签（条内右侧）
        ax.text(val - 2, y[i], f"{val:.1f}%",
                ha="right", va="center",
                fontsize=FS_MAIN+0.5, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=1.5, foreground=cm)])

        # 轴标签（左侧）
        ax.text(-1.5, y[i], lbl,
                ha="right", va="center", fontsize=FS_ANNOT,
                color=C_NAVY, fontweight="bold",
                multialignment="right")

    # OA 文字摘要
    ax.text(50, 0.04,
            f"Overall Accuracy = {OA:.1f}%",
            ha="center", va="bottom", fontsize=FS_ANNOT,
            color=C_NEUTRAL, style="italic")

    ax.set_xlim(-0.5, 106)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Rate (%)", fontsize=FS_MAIN, labelpad=4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _draw_cm(ax):
    """b: 混淆矩阵 — 精心配色 + 清晰单元格"""
    cm_pct = CM / CM.sum() * 100

    # 自定义 colormap：白→浅蓝→深蓝
    cmap = LinearSegmentedColormap.from_list(
        "naval", ["#FFFFFF", "#C8DDEF", "#2166AC"], N=256)

    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=60, aspect="auto")

    # 单元格内容
    cell_info = [
        [("Hit", "TP", C_TEAL),    ("False Alarm", "FP", C_CORAL)],
        [("Miss",  "FN", C_AMBER), ("Correct Rej.", "TN", C_BLUE)],
    ]

    for i in range(2):
        for j in range(2):
            bg_dark = cm_pct[i, j] > 28
            txt_c = "white" if bg_dark else "#222222"
            lbl, tag, accent = cell_info[i][j]

            # 圆角标签徽章
            badge_c = "rgba(255,255,255,0.25)" if bg_dark else accent
            badge_fc = (1,1,1,0.25) if bg_dark else (*mpl.colors.to_rgb(accent), 0.12)

            # 主数字
            ax.text(j, i - 0.15, f"{CM[i,j]:,}",
                    ha="center", va="center",
                    fontsize=FS_MAIN+0.5, fontweight="bold", color=txt_c)
            # 百分比
            ax.text(j, i + 0.15, f"({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=FS_ANNOT, color=txt_c, alpha=0.85)
            # 类型标签
            lbl_c = "white" if bg_dark else accent
            ax.text(j, i + 0.42, f"{lbl}  [{tag}]",
                    ha="center", va="center",
                    fontsize=FS_SMALL, color=lbl_c,
                    style="italic", alpha=0.9)

    # Colorbar
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.042, pad=0.03,
                             orientation="vertical")
    cb.set_label("Proportion (%)", fontsize=FS_SMALL, labelpad=3)
    cb.ax.tick_params(labelsize=FS_SMALL, length=2)
    cb.outline.set_linewidth(0.4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MODIS Cloud", "MODIS Clear"],
                       fontsize=FS_ANNOT, color="#444")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["L2 Cloud", "L2 Clear"],
                       fontsize=FS_ANNOT, color="#444")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0)

    # 轴标题
    # ax.set_xlabel("MODIS Reference", fontsize=FS_ANNOT, labelpad=6, color="#555")
    # ax.set_ylabel("AGRI L2 Prediction", fontsize=FS_ANNOT, labelpad=6, color="#555")


def _draw_stratified_oa(ax):
    """c: 分层 OA 柱状图 — 渐变柱 + 精致标注"""
    ax.set_facecolor("white")

    strata = [
        ("Overall", OA, C_BLUE, "#92BFE0"),
        ("High CF\n(MODIS > 60%)", OA_HIGH, C_TEAL, "#90D4C8"),
        ("Low CF\n(MODIS ≤ 60%)", OA_LOW, C_AMBER, "#F0CF90"),
    ]
    x = np.arange(len(strata))

    # 水平参考网格（极淡）
    for yg in [20, 40, 60, 80, 100]:
        ax.axhline(yg, color="#EEEEEE", lw=0.6, zorder=0)

    bar_w = 0.46
    for xi, (lbl, val, c_top, c_bot) in enumerate(strata):
        # 渐变柱（用 imshow 模拟）
        n_grad = 100
        grad_data = np.linspace(0, 1, n_grad).reshape(n_grad, 1)
        grad_cmap = LinearSegmentedColormap.from_list("g", [c_top, c_bot])
        ax.imshow(grad_data, cmap=grad_cmap, aspect="auto",
                  extent=[xi - bar_w/2, xi + bar_w/2, 0, val],
                  origin="lower", zorder=2, alpha=0.92)
        # 边框
        rect = mpatches.FancyBboxPatch(
            (xi - bar_w/2, 0), bar_w, val,
            boxstyle="square,pad=0",
            facecolor="none", edgecolor=c_top, linewidth=0.7, zorder=3)
        ax.add_patch(rect)

        # 数值标签
        ax.text(xi, val + 2.2, f"{val:.1f}%",
                ha="center", fontsize=FS_MAIN, fontweight="bold",
                color=c_top, zorder=4)

        # x 轴标签
        ax.text(xi, -8, lbl, ha="center", va="top",
                fontsize=FS_ANNOT, color="#444",
                multialignment="center")

    # 高/低CF之差标注
    delta = OA_HIGH - OA_LOW
    ax.annotate("", xy=(2, OA_LOW), xytext=(2, OA_HIGH),
                arrowprops=dict(arrowstyle="<->", color=C_NEUTRAL,
                                lw=0.8, mutation_scale=6))
    ax.text(2.28, (OA_HIGH + OA_LOW)/2, f"Δ {delta:.1f} pp",
            va="center", fontsize=FS_SMALL, color=C_NEUTRAL, style="italic")

    ax.set_xlim(-0.6, 2.9)
    ax.set_ylim(0, 108)
    ax.set_xticks([])
    ax.set_ylabel("Pooled OA (%)", fontsize=FS_MAIN, labelpad=4)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_ANNOT)


def _draw_cf_scatter(ax):
    """d: 云量散点图 — 精致配色 + 尺寸图例"""
    ax.set_facecolor("#FAFBFC")

    l2_cf = [s["l2_cf"] for s in SCENES]
    cm_cf = [s["cm_cf"] for s in SCENES]
    oa_v  = [s["oa"]    for s in SCENES]
    ns    = [s["n"]     for s in SCENES]

    # 尺寸映射（更显眼）
    sizes = [max(45, s["n"] / 6500) for s in SCENES]
    edge_lw = [1.5 if s["cm_cf"] > 60 else 0.5 for s in SCENES]
    edge_c  = [C_TEAL if s["cm_cf"] > 60 else "#BBBBBB" for s in SCENES]

    # y=x 参考线
    ax.plot([38, 98], [38, 98], ls="--", color="#CCCCCC", lw=0.8,
            zorder=0, label="1:1 line")
    ax.fill_between([38, 98], [38, 98], [98, 98],
                    color="#EBF4F0", alpha=0.35, zorder=0)
    ax.fill_between([38, 98], [38, 38], [38, 98],
                    color="#F5EBE8", alpha=0.35, zorder=0)
    ax.text(90, 92, "L2 < MODIS", fontsize=FS_SMALL, color="#8BC4B0",
            ha="center", style="italic", va="center")
    ax.text(90, 48, "L2 > MODIS", fontsize=FS_SMALL, color="#E0A898",
            ha="center", style="italic", va="center")

    # 自定义配色 cmap
    cmap_sc = LinearSegmentedColormap.from_list(
        "oa", ["#D62728", "#FF9500", "#FEDC5A", "#66C55C", "#1A7030"], N=256)

    sc = ax.scatter(l2_cf, cm_cf, c=oa_v, cmap=cmap_sc, vmin=38, vmax=92,
                    s=sizes, edgecolors=edge_c, linewidths=edge_lw,
                    zorder=3, alpha=0.92)

    # 时间标签
    # for s in SCENES:
    #     ax.annotate(s["ts"], (s["l2_cf"], s["cm_cf"]),
    #                 textcoords="offset points", xytext=(5, 3),
    #                 fontsize=4.5, color="#666666")

    # colorbar
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("OA (%)", fontsize=FS_ANNOT, labelpad=3)
    cb.ax.tick_params(labelsize=FS_SMALL, length=2)
    cb.outline.set_linewidth(0.4)

    # 尺寸图例（代表像素数）
    # for n_ref, lbl in [(100000, "100k"), (400000, "400k")]:
    #     s_ref = max(45, n_ref / 6500)
    #     ax.scatter([], [], s=s_ref, c="#AAAAAA", alpha=0.7,
    #                edgecolors="#888", linewidths=0.5, label=f"n = {lbl}")
    # ax.legend(fontsize=FS_SMALL, loc="lower right",
    #           handletextpad=0.4, labelspacing=0.5,
    #           borderpad=0.6, frameon=True,
    #           framealpha=0.85, edgecolor="#DDDDDD", fancybox=True)

    ax.set_xlabel("AGRI L2 Cloud Fraction (%)", fontsize=FS_MAIN, labelpad=4)
    ax.set_ylabel("MODIS CM Cloud Fraction (%)", fontsize=FS_MAIN, labelpad=4)
    ax.set_xlim(40, 96)
    ax.set_ylim(40, 96)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FS_ANNOT)


if __name__ == "__main__":
    main()