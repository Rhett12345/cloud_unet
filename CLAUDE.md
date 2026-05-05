# AGRI + MYD06 Cloud Property Retrieval Pipeline

## Environment

Always use the `cloudunet` conda environment:

```bash
conda run -n cloudunet python <script>
# or activate first:
conda activate cloudunet
```

GPU: 2× NVIDIA GeForce RTX 4090. Set `CUDA_VISIBLE_DEVICES=1` if GPU 0 is occupied.

Matplotlib rendering may require:
```bash
LD_LIBRARY_PATH=/home/yuq/anaconda3/envs/cloudunet/lib:$LD_LIBRARY_PATH MPLCONFIGDIR=/tmp/matplotlib
```

## Project Structure

```
unet/
├── config.py          # Global config: paths, hyperparams, date splits
├── fusion_config.py   # Fusion QC thresholds (env-overridable)
├── fusion_core.py     # MODIS→AGRI aggregation engine (pure numeric, no IO)
├── fusion_io.py       # File IO: AGRI/MYD06/MYD03 reads, H5 writes, QC filters
├── data_fusion.py     # Fusion scheduler: multiprocess orchestration
├── dataset.py         # PyTorch Dataset + stats computation
├── sample_filters.py  # Patch/sample supervision filtering
├── model.py           # CloudPropertyNet (ConvNeXt + DA + Transformer U-Net)
├── train.py           # Training loop with AMP, multi-checkpoint saving
├── test.py            # Evaluation: CLP OA, per-class acc, regression metrics
├── main.py            # Pipeline orchestrator (fuse/stats/train/test/infer)
├── tools/             # Diagnostic and utility scripts
│   ├── eval_checkpoints.py
│   ├── analyze_qc_failures.py
│   └── balance_split.py
├── tests/             # Unit tests
├── scripts/           # Diagnostic scripts (visualization, quality, etc.)
├── runs/              # QC diagnostics output
├── logs/              # Training logs
└── summary/           # Session summaries
```

## Pipeline Stages

1. **fuse** — MODIS (MYD06/MYD03) → AGRI grid fusion, outputs H5 samples
2. **stats** — Compute normalization statistics from train split
3. **train** — Train CloudPropertyNet
4. **test** — Evaluate on test split
5. **infer** — Full-disk inference on new AGRI scenes

## Key Commands

```bash
# Full pipeline
conda run -n cloudunet python main.py --stages fuse stats train test

# Fusion only, one day
conda run -n cloudunet python data_fusion.py --split train --day 20190105 --workers 8

# Training only
conda run -n cloudunet python main.py --stages train

# Test a specific checkpoint
conda run -n cloudunet python test.py --checkpoint unet_workdir/model/HIR_COMP_UNet_AGRIonly_best_oa.pth

# Run unit tests
conda run -n cloudunet python -m pytest -p no:cacheprovider tests

# QC diagnostics (single day, single process)
ENABLE_QC_DIAGNOSTICS=true conda run -n cloudunet python data_fusion.py \
  --split train --day 20190105 --workers 1 --enable-qc-diagnostics

# Check date split balance
conda run -n cloudunet python tools/balance_split.py --report

# Suggest stratified date split
conda run -n cloudunet python tools/balance_split.py --suggest --seed 42
```

## Important Env Vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNET_WORKDIR` | `/data/Data_yuq/unet_workdir` | Root output dir |
| `UNET_CHECKPOINT_MONITOR` | `val_macro_acc` | Checkpoint selection metric |
| `UNET_TRAIN_DATES` | config list | Override train dates |
| `UNET_VAL_DATES` | config list | Override val dates |
| `UNET_TEST_DATES` | config list | Override test dates |
| `FUSION_AGRI_SEARCH_RADIUS_KM` | `2.5` | MODIS→AGRI match radius |
| `FUSION_REG_TIME_MAX_MIN` | `3.0` | Max time diff for regression labels |
| `ENABLE_QC_DIAGNOSTICS` | `false` | Enable per-scene QC gate CSV/JSONL |

## Design Conventions

- Do not change model structure or fusion thresholds unless explicitly requested
- Keep changes scoped; prefer editing existing files over creating new ones
- New diagnostics/scripts go under `tools/`
- Prefer CSV/JSON outputs for diagnostics
- Put new fusion outputs under separate experiment dirs, don't overwrite old results
- Run `pytest -p no:cacheprovider tests` after changes to verify nothing broke

 每次回复前先说"打报告"。
  /home/yuq/cloudmask/GeoISCLD-Net/路径是原始代码路径可供参考，但是我们用的数据不一样，所以只能够参考