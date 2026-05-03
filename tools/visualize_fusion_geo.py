"""
visualize_fusion_geo.py
=======================
Visualize fused H5 geometry: AGRI disk boundary, MODIS label coverage,
lat/lon match quality.  Use to verify MODIS data falls within the AGRI disk.

Usage:
  # Single file (full_disk or samples format)
  conda run -n cloudunet python tools/visualize_fusion_geo.py /path/to/AGRI_MYD06_*.h5

  # All files in a day dir
  conda run -n cloudunet python tools/visualize_fusion_geo.py /data/.../paired/train/20190105/

  # Multiple files, save figures instead of showing
  conda run -n cloudunet python tools/visualize_fusion_geo.py /path/to/dir/ --save-dir /tmp/geo_viz/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on path for config import
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon

import config as cfg

# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

CLP_NAMES = getattr(cfg, "CLP_CLASS_NAMES", ["Clear", "Water", "Ice"])
CLP_CMAP = ListedColormap(["white", "deepskyblue", "red"][: len(CLP_NAMES)])
CLP_NORM = BoundaryNorm(np.arange(len(CLP_NAMES) + 1) - 0.5, len(CLP_NAMES))


def _read_full_disk(h5_path: Path) -> Optional[dict]:
    """Read old-format full_disk H5, return dict of arrays or None."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "Labels" not in f or "AGRI" not in f:
                return None
            lat = f["AGRI/Geolocation/lat"][()].astype(np.float32)
            lon = f["AGRI/Geolocation/lon"][()].astype(np.float32)
            clp = f["Labels/CLP"][()].astype(np.float32)
            cer = f["Labels/CER"][()] if "CER" in f["Labels"] else None
            dt = None
            ovlp = None
            if "QA" in f:
                dt = f["QA/MATCH_DT_MIN"][()] if "MATCH_DT_MIN" in f["QA"] else None
                ovlp = (
                    f["QA/OVERLAP_FRACTION"][()] if "OVERLAP_FRACTION" in f["QA"] else None
                )
            return dict(lat=lat, lon=lon, clp=clp, cer=cer, dt=dt, ovlp=ovlp)
    except Exception as exc:
        print(f"  [WARN] {h5_path.name}: {exc}")
        return None


def _read_samples(h5_path: Path) -> Optional[dict]:
    """Read new-format samples_v2 H5, return dict of arrays or None."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "Samples" not in f:
                return None
            s = f["Samples"]
            if "geo" not in s or "labels" not in s:
                return None
            lat = s["geo"][:, 0].astype(np.float32)  # (N, H, W) -> take first channel
            lon = s["geo"][:, 1].astype(np.float32)
            clp = s["labels"][:, 0].astype(np.float32)
            cer = s["labels"][:, 1].astype(np.float32) if s["labels"].shape[1] > 1 else None
            dt = s["max_time_diff_min"][()] if "max_time_diff_min" in s else None
            ovlp = s["mean_overlap_frac"][()] if "mean_overlap_frac" in s else None
            return dict(
                lat=lat.reshape(-1),
                lon=lon.reshape(-1),
                clp=clp.reshape(-1),
                cer=cer.reshape(-1) if cer is not None else None,
                dt=dt,
                ovlp=ovlp,
                is_patches=True,
            )
    except Exception as exc:
        print(f"  [WARN] {h5_path.name}: {exc}")
        return None


def read_fused(h5_path: Path) -> Optional[dict]:
    """Auto-detect format and read."""
    data = _read_samples(h5_path)
    if data is not None:
        return data
    return _read_full_disk(h5_path)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_disk_outline(ax, lat, lon, **kwargs):
    """Overlay the AGRI valid-pixel boundary (convex hull of valid lat/lon)."""
    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 3:
        return
    # Sample boundary points: take extreme lat/lon in azimuthal bins
    n_bins = 72
    y = lat[valid]
    x = lon[valid]
    center_lat, center_lon = np.median(y), np.median(x)
    angles = np.arctan2(y - center_lat, x - center_lon)
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hull_lat, hull_lon = [], []
    for i in range(n_bins):
        mask = (angles >= bins[i]) & (angles < bins[i + 1])
        if mask.sum() == 0:
            continue
        # furthest point from center in this bin
        dist = np.sqrt((y[mask] - center_lat) ** 2 + (x[mask] - center_lon) ** 2)
        idx = np.argmax(dist)
        hull_lat.append(y[mask][idx])
        hull_lon.append(x[mask][idx])
    if len(hull_lat) < 3:
        return
    hull_lat = np.array(hull_lat)
    hull_lon = np.array(hull_lon)
    # Sort by angle around centroid
    order = np.argsort(np.arctan2(hull_lat - center_lat, hull_lon - center_lon))
    ax.plot(
        hull_lon[order],
        hull_lat[order],
        **kwargs,
    )
    ax.plot(
        [hull_lon[order[-1]], hull_lon[order[0]]],
        [hull_lat[order[-1]], hull_lat[order[0]]],
        **kwargs,
    )


def _plot_agri_valid(ax, lat, lon, **kwargs):
    """Scatter plot of valid AGRI pixel centroids (subsampled)."""
    valid = np.isfinite(lat) & np.isfinite(lon)
    y, x = lat[valid], lon[valid]
    step = max(1, len(y) // 5000)
    ax.scatter(x[::step], y[::step], s=0.3, alpha=0.5, **kwargs)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def plot_single(data: dict, title: str, save_path: Optional[Path] = None):
    """Create a 2×2 diagnostic figure for one fused scene."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax_map, ax_clp), (ax_dt, ax_dist) = axes

    is_patches = data.get("is_patches", False)
    lat = data["lat"]
    lon = data["lon"]
    clp = data["clp"]
    cer = data.get("cer")
    dt = data.get("dt")
    ovlp = data.get("ovlp")

    # ── Map: AGRI disk outline + MODIS CLP coverage ──
    _plot_disk_outline(ax_map, lat, lon, color="blue", lw=1.5, label="AGRI disk boundary")
    clp_valid = np.isfinite(clp) & (clp >= 0)
    if clp_valid.any():
        y, x = lat[clp_valid], lon[clp_valid]
        step = max(1, len(y) // 8000)
        ax_map.scatter(
            x[::step], y[::step], c=clp[clp_valid][::step],
            cmap=CLP_CMAP, norm=CLP_NORM, s=0.8, alpha=0.7, rasterized=True,
        )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title(f"CLP coverage\n{title}")
    ax_map.legend(loc="lower right", fontsize=7)
    ax_map.set_aspect("equal")

    # ── CLP histogram ──
    if clp_valid.any():
        counts = np.bincount(
            clp[clp_valid].astype(int), minlength=len(CLP_NAMES)
        )[: len(CLP_NAMES)]
        bars = ax_clp.bar(range(len(CLP_NAMES)), counts, color=CLP_CMAP.colors)
        for b, c in zip(bars, counts):
            ax_clp.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.02,
                       str(c), ha="center", fontsize=8)
    ax_clp.set_xticks(range(len(CLP_NAMES)))
    ax_clp.set_xticklabels(CLP_NAMES, fontsize=8)
    ax_clp.set_ylabel("Pixel count")
    coverage = 100.0 * clp_valid.mean() if len(clp) > 0 else 0
    ax_clp.set_title(f"Phase distribution  (valid={coverage:.1f}%)")

    # ── Time diff histogram ──
    if dt is not None and len(dt) > 0:
        dt_valid = dt[np.isfinite(dt)]
        if len(dt_valid) > 0:
            ax_dt.hist(dt_valid, bins=30, color="steelblue", edgecolor="white")
            ax_dt.axvline(np.median(dt_valid), color="red", ls="--", label=f"median={np.median(dt_valid):.1f}min")
            ax_dt.legend(fontsize=7)
    ax_dt.set_xlabel("MATCH_DT_MIN (min)")
    ax_dt.set_ylabel("Count")
    if is_patches:
        ax_dt.set_title("Per-patch max time diff")
    else:
        ax_dt.set_title("Per-pixel time diff")

    # ── MATCH_DIST or overlap hist ──
    if ovlp is not None and len(ovlp) > 0:
        ovlp_valid = ovlp[np.isfinite(ovlp)]
        if len(ovlp_valid) > 0:
            ax_dist.hist(ovlp_valid, bins=20, color="tomato", edgecolor="white")
            ax_dist.axvline(np.median(ovlp_valid), color="blue", ls="--", label=f"median={np.median(ovlp_valid):.2f}")
            ax_dist.legend(fontsize=7)
        ax_dist.set_xlabel("Overlap fraction" if not is_patches else "Mean overlap fraction")
        ax_dist.set_title("Spatial coverage")
    else:
        ax_dist.text(0.5, 0.5, "No overlap data", ha="center", va="center",
                    transform=ax_dist.transAxes)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_summary(all_data: List[dict], titles: List[str], save_path: Path):
    """Multi-file summary: lat/lon overlay of all scenes + per-file stats."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_map, ax_bar = axes

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_data), 1)))

    # ── Map overlay: all scenes' CLP coverage ──
    for i, (data, title) in enumerate(zip(all_data, titles)):
        lat, lon, clp = data["lat"], data["lon"], data["clp"]
        clp_valid = np.isfinite(clp) & (clp >= 0)
        if not clp_valid.any():
            continue
        y, x = lat[clp_valid], lon[clp_valid]
        step = max(1, len(y) // 3000)
        ax_map.scatter(
            x[::step], y[::step], s=0.4, alpha=0.5, color=colors[i],
            label=title, rasterized=True,
        )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title(f"Multi-scene CLP coverage ({len(all_data)} scenes)")
    ax_map.legend(fontsize=6, loc="lower right", markerscale=5)
    ax_map.set_aspect("equal")

    # ── Per-scene stats bar ──
    stats = []
    for data, title in zip(all_data, titles):
        clp = data["clp"]
        clp_valid = np.isfinite(clp) & (clp >= 0)
        cloudy = clp_valid & (clp > 0)
        stats.append({
            "scene": title,
            "valid_pct": 100.0 * clp_valid.mean() if len(clp) > 0 else 0,
            "cloudy_pct": 100.0 * cloudy.sum() / max(clp_valid.sum(), 1) if clp_valid.any() else 0,
        })

    x_pos = np.arange(len(stats))
    ax_bar.bar(x_pos - 0.15, [s["valid_pct"] for s in stats], 0.3, color="steelblue", label="Valid CLP %")
    ax_bar.bar(x_pos + 0.15, [s["cloudy_pct"] for s in stats], 0.3, color="tomato", label="Cloudy % of valid")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([s["scene"] for s in stats], rotation=45, ha="right", fontsize=6)
    ax_bar.set_ylabel("%")
    ax_bar.set_title("Per-scene label coverage")
    ax_bar.legend(fontsize=7)

    fig.suptitle(f"Fusion geometry summary — {len(all_data)} scenes", fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Summary saved -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize fused H5 geometry")
    parser.add_argument("inputs", nargs="+", help="H5 files or directories")
    parser.add_argument("--save-dir", default=None, help="Save figures here instead of showing")
    parser.add_argument("--max-files", type=int, default=20, help="Max files to process")
    parser.add_argument("--summary-only", action="store_true", help="Only generate multi-file summary")
    args = parser.parse_args()

    h5_files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            h5_files.extend(sorted(p.rglob("*.h5")))
        elif p.is_file() and p.suffix == ".h5":
            h5_files.append(p)

    if not h5_files:
        print("No .h5 files found.")
        sys.exit(1)

    h5_files = h5_files[: args.max_files]
    print(f"Processing {len(h5_files)} files...")

    all_data = []
    titles = []
    save_dir = Path(args.save_dir) if args.save_dir else None

    for i, h5f in enumerate(h5_files):
        print(f"  [{i+1}/{len(h5_files)}] {h5f.name}")
        data = read_fused(h5f)
        if data is None:
            print(f"    [SKIP] could not read")
            continue

        all_data.append(data)
        # Short title: YYYYMMDD_HHMMSS or first 24 chars of filename
        stem = h5f.stem
        if len(stem) > 24:
            stem = stem[:24]
        titles.append(stem)

        if not args.summary_only and save_dir:
            plot_single(data, stem, save_dir / f"{stem}_geo.png")

    if not all_data:
        print("No readable data found.")
        sys.exit(1)

    if save_dir:
        plot_summary(all_data, titles, save_dir / "summary_geo.png")
    else:
        print(f"Read {len(all_data)} files. Use --save-dir to output figures.")
        # Print quick stats
        for title, data in zip(titles, all_data):
            clp_valid = np.isfinite(data["clp"]) & (data["clp"] >= 0)
            print(f"  {title}: valid_clp={clp_valid.sum()} ({100*clp_valid.mean():.1f}%)")


if __name__ == "__main__":
    main()
