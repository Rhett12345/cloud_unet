# AGRI + MYD06 Cloud Property Retrieval Pipeline

Retrieves cloud properties (CLP, CTH) from FY-4A AGRI FDI/GEO data,
supervised by MYD06 (Aqua/MODIS) labels with MYD03 1km geolocation,
using a U-Net architecture.

---

## Project layout

```
.
├── config.py          ← ALL paths and hyper-parameters (edit here only)
├── fusion_config.py   ← Fusion thresholds: time window, match radius, QC gates
├── fusion_core.py     ← MODIS→AGRI aggregation engine (pure numeric, no IO)
├── fusion_io.py       ← AGRI / MYD06 / MYD03 file IO, QC filters, HDF5 write
├── data_fusion.py     ← Stage 1: multi-process fusion scheduler
├── sample_filters.py  ← Patch / sample supervision quality filters
├── dataset.py         ← Stage 2: PyTorch Dataset + normalisation statistics
├── model.py           ← U-Net architecture (AGRI + geo channels)
├── train.py           ← Stage 3: training loop with AMP, multi-checkpoint saving
├── test.py            ← Stage 4: evaluation (CLP OA, per-class acc, CTH regression)
├── inference.py       ← Stage 5: full-disk sliding-window inference
├── main.py            ← Orchestrator (single entry point for all stages)
├── validate_modis.py  ← MODIS CTH validation against AGRI L2 / model predictions
├── requirements.txt
├── tools/             ← Diagnostic and utility scripts
│   ├── balance_split.py          ← Date split balance analysis
│   ├── baseline_l2_vs_modis.py   ← AGRI L2 vs MODIS baseline comparison
│   ├── geoloc_offset_diag.py     ← AGRI–MODIS geolocation offset diagnostics
│   ├── visualize_fusion_geo.py   ← Fusion geolocation matching visualization
│   └── visualize_h5_paired_batch.py ← Paired HDF5 batch visualization
├── tests/             ← Unit tests
├── runs/              ← QC diagnostics output
├── logs/              ← Training logs
└── summary/           ← Session summaries
```

---

## Quick start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Environment

Always use the `cloudunet` conda environment:

```bash
conda activate cloudunet
```

GPU: 2× NVIDIA GeForce RTX 4090. Set `CUDA_VISIBLE_DEVICES=1` if GPU 0 is occupied.

Matplotlib rendering may require:

```bash
LD_LIBRARY_PATH=/home/yuq/anaconda3/envs/cloudunet/lib:$LD_LIBRARY_PATH MPLCONFIGDIR=/tmp/matplotlib
```

### 3 · Edit config.py

Set at minimum the data paths (defaults point to `/data/Data_yuq/`):

```python
AGRI_ROOT    = Path("/your/AGRI/data")     # parent of YYYYMMDD/ day-folders
MODIS_ROOT   = Path("/your/MYD06/data")    # parent of YYYYMMDD/ day-folders
FY4A_L2_ROOT = Path("/your/AGRI_L2/data")  # AGRI L2 products (CTH etc.)
ROOT         = Path("/your/workdir")       # all outputs written here
```

Date splits can be overridden via env vars `UNET_TRAIN_DATES`, `UNET_VAL_DATES`, `UNET_TEST_DATES`.

### 4 · Run the full pipeline

```bash
conda run -n cloudunet python main.py --stages fuse stats train test
```

Or step by step:

```bash
# Fuse one day of training data
conda run -n cloudunet python main.py --stages fuse --split train --day 20190105 --workers 8

# Compute normalisation statistics from train split
conda run -n cloudunet python main.py --stages stats

# Train the model
conda run -n cloudunet python main.py --stages train

# Evaluate on test split
conda run -n cloudunet python main.py --stages test

# Full-disk inference on new AGRI scenes
conda run -n cloudunet python main.py --stages infer --agri_dir /data/FY4A/20190105/
```

---

## Data directory structure expected

```
AGRI_ROOT/
  20190105/
    FY4A-_AGRI--_N_DISK_xxx_20190105060000_L1.HDF      (FDI)
    FY4A-_AGRI--_N_DISK_xxx_20190105060000_L1_GEO.HDF   (GEO)
    ...

MODIS_ROOT/
  20190105/
    MYD06_L2.A2019005.0600.061.*.hdf    (includes MYD03 geolocation)
    ...

FY4A_L2_ROOT/
  CTH/
    20190105/
      FY4A-_AGRI--_N_DISK_xxx_CTH.HDF   (AGRI L2 CTH products)
      ...
```

---

## Output structure

```
ROOT/
  paired/
    train/<YYYYMMDD>/AGRI_MYD06_pair_YYYYMMDD_HHMMSS.h5
    val/  ...
    test/ ...
  stats/
    norm_stats.npz
  model/
    HIR_COMP_UNet_AGRIonly_best.pth        (best by monitored metric)
    HIR_COMP_UNet_AGRIonly_best_loss.pth   (best by val loss)
    HIR_COMP_UNet_AGRIonly_best_oa.pth     (best by overall accuracy)
    HIR_COMP_UNet_AGRIonly_best_macro.pth  (best by macro accuracy)
    HIR_COMP_UNet_AGRIonly_last.pth        (most recent epoch)
  logs/
    pipeline.log
    train_log.csv
  retrieval/
    <stem>_retrieval.npz   (lat, lon, CLP_pred, CTH_pred, CLP_prob)
  eval/
    metrics_summary.csv
    confusion_matrix.png
    scatter_CTH.png
```

---

## Paired HDF5 format (produced by data_fusion.py)

Default mode is `samples_v2` — each HDF5 file contains a collection of patches:

```
/Samples/agri             float32 (N, 6, 64, 64)   AGRI BT patches (6 IR channels)
/Samples/geo              float32 (N, 4, 64, 64)   lat, lon, VZA, SZA
/Samples/labels           float32 (N, 2, 64, 64)   [CLP, CTH]
/Samples/row / col         int32  (N,)              patch origin in scene
/Samples/valid_clp_px      int32  (N,)              valid CLP pixels in patch
/Samples/valid_cloudy_px   int32  (N,)              valid cloudy pixels in patch
/Samples/valid_{clear,water,ice}_px  int32  (N,)    per-class pixel counts
/Samples/max_time_diff_min float32 (N,)             max AGRI–MODIS Δt in patch
/Samples/p95_match_dist_km float32 (N,)             P95 match distance
/Samples/mean_{overlap_frac,sample_weight,cloud_frac,phase_consist}  float32 (N,)
```

File-level attributes: `format`, `agri_datetime`, `agri_channels`, `patch_size`, `mode`, `clp_class_names`, `time_low_q_min`, `reg_time_max_min`, `num_samples`.

Legacy format (full-disk `AGRI/BT` + `Labels/*`, mode=`full_disk`) is also supported for reading.

---

## Model input / output

|         | Channels | Description |
|---------|----------|-------------|
| Input   | 6        | AGRI BT ch9–ch14 (indices 8–13) |
| Input   | 4        | Geo: lat, lon, VZA, SZA |
| Output  | 3        | CLP logits (Clear / Water / Ice) |
| Output  | 1        | CTH (cloud top height, m, normalised) |

Architecture: U-Net with 5 encoder stages (DoubleConv + MaxPool), bottleneck, 4 decoder stages (bilinear upsample + skip connection), single 1×1 conv head. Base channels: 64.

---

## Label channels in model output

| Channel | Variable | Unit      | Loss                          |
|---------|----------|-----------|-------------------------------|
| 0       | CLP      | class     | CrossEntropy (3-class)        |
| 1       | CTH      | m         | SmoothL1, z-score normalised  |

CLP class mapping: 0=Clear, 1=Water, 2=Ice. CER and COT have been removed from the pipeline.

---

## Key hyper-parameters (all in config.py)

| Parameter            | Default   | Description                         |
|----------------------|-----------|-------------------------------------|
| AGRI_BT_CHANNEL_INDICES | [8..13] | 6 IR channels (0-based indices)     |
| PATCH_SIZE           | (64, 64)  | Training patch size                 |
| BATCH_SIZE           | 64        | Training batch size                 |
| NUM_EPOCHS           | 30        | Training epochs                     |
| LEARNING_RATE        | 1e-4      | AdamW initial LR                    |
| UNET_BASE_CHANNELS   | 64        | UNet base width                     |
| LOSS_W_CLP           | 1.0       | CLP classification loss weight      |
| LOSS_W_CTH           | 1.0       | CTH regression loss weight          |
| GRAD_CLIP            | 1.0       | Gradient clipping max norm          |
| LR_PATIENCE          | 6         | ReduceLROnPlateau patience (epochs) |
| EARLY_STOP_PATIENCE  | 10        | Early stopping patience (epochs)    |
| RANDOM_SEED          | 42        | Reproducibility seed                |

---

## Fusion parameters (fusion_config.py, env-overridable)

| Parameter               | Default | Description                         |
|-------------------------|---------|-------------------------------------|
| TIME_LOW_Q_MIN          | 7.5     | Max AGRI–MODIS Δt for CLP (min)     |
| REG_TIME_MAX_MIN        | 7.5     | Max AGRI–MODIS Δt for CTH (min)     |
| AGRI_SEARCH_RADIUS_KM   | 2.5     | MODIS→AGRI KD-tree match radius     |
| MAX_VZA_DEG / MAX_SZA_DEG | 65    | Satellite / solar zenith angle max  |
| MAX_CTH_M               | 18000   | Maximum valid CTH (m)               |
| MAX_MATCH_DIST_KM       | 3.0     | Max MODIS→AGRI spatial distance     |

---

## Date split (2019 full year)

| Split | Dates |
|-------|-------|
| Train | 1st & 25th of each month (24 days) |
| Val   | 15th of odd months (6 days) |
| Test  | 15th of even months (6 days) |

---

## Key env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNET_WORKDIR` | `/data/Data_yuq/unet_workdir` | Root output directory |
| `UNET_CHECKPOINT_MONITOR` | `val_macro_acc` | Metric for best checkpoint |
| `UNET_TRAIN_DATES` | config list | Override train dates |
| `UNET_VAL_DATES` | config list | Override val dates |
| `UNET_TEST_DATES` | config list | Override test dates |
| `UNET_LOSS_W_CLP` | `1.0` | CLP loss weight |
| `UNET_LOSS_W_CTH` | `1.0` | CTH loss weight |
| `FUSION_AGRI_SEARCH_RADIUS_KM` | `2.5` | MODIS→AGRI match radius |
| `FUSION_REG_TIME_MAX_MIN` | `7.5` | Max time diff for regression |
| `ENABLE_QC_DIAGNOSTICS` | `false` | Per-scene QC gate CSV/JSONL output |
| `CUDA_VISIBLE_DEVICES` | `1` | GPU selection (if GPU 0 busy) |

---

## Running tests

```bash
conda run -n cloudunet python -m pytest -p no:cacheprovider tests
```

---

## Diagnostic tools

```bash
# Check date split balance
conda run -n cloudunet python tools/balance_split.py --report

# Suggest stratified date split
conda run -n cloudunet python tools/balance_split.py --suggest --seed 42

# Visualize fusion geo matching
conda run -n cloudunet python tools/visualize_fusion_geo.py

# QC diagnostics (single day, single process)
ENABLE_QC_DIAGNOSTICS=true conda run -n cloudunet python data_fusion.py \
  --split train --day 20190105 --workers 1 --enable-qc-diagnostics

# Geolocation offset diagnostics (AGRI vs MODIS spatial alignment)
conda run -n cloudunet python tools/geoloc_offset_diag.py --day 20190405
conda run -n cloudunet python tools/geoloc_offset_diag.py --days 20190401 20190405 20190410

# AGRI L2 vs MODIS baseline comparison
conda run -n cloudunet python tools/baseline_l2_vs_modis.py
```
