# pig-weight

Non-contact pig weight estimation from 2D RGB imagery (PIGRGB-Weight
dataset). This repo currently covers **Phase 1: Segmentation** — training
an encoder-swappable U-Net to isolate the pig/dorsal region, evaluated
against ground-truth masks and a zero-shot SAM2 baseline.

See `CLAUDE.md` for the dataset's ground-truth structure and the locked
design decisions, and `docs/superpowers/specs/2026-09-10-phase1-segmentation-design.md`
for the full design spec.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`sam2` installs from GitHub (see `requirements.txt`) — it has no PyPI
release. You'll also need a SAM2 checkpoint for the baseline:

```bash
mkdir -p checkpoints/sam2
curl -L -o checkpoints/sam2/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

## Data

Place the extracted `PigRGB-Weight/` dataset under `data/` (gitignored —
not tracked in this repo). Then generate the manifest and split files
(paths configured in `configs/segmentation.yaml`):

```bash
python audit_pigrgb.py                              # produces data/manifest.csv
python -m src.data.splits                            # produces data/splits.csv
```

`audit_pigrgb.py` parses the dataset's folder structure into the manifest
schema documented in the design spec and in `src/data/splits.py`'s
docstring, and checks pig-id/mask-pairing integrity.

## Running

```bash
# Train the segmentation U-Net
python -m src.training.train --config configs/segmentation.yaml

# Resume from a checkpoint
python -m src.training.train --config configs/segmentation.yaml \
  --resume checkpoints/segmentation/last.pt

# Zero-shot SAM2 baseline (comparison point, no training)
python -m src.evaluation.sam2_baseline --config configs/segmentation.yaml --split test

# Tests
pytest tests/ -v
```

Swap the segmentation encoder by editing `configs/segmentation.yaml`'s
`model.encoder_name` (`vanilla`, `timm-mobilenetv3_small_100`, `resnet34`,
...) — no code changes needed.

Device selection is config-driven (`device: auto|cuda|mps|cpu` in the
config, or `--device` on the CLI). This Mac (Apple M2) is dev/scaffolding
only — no CUDA, MPS backend available; the actual training machine is TBD.

## Two intentionally-stubbed components

Left as `# TODO(timotius)` for hands-on implementation, with tests written
against the stub signatures (they currently fail with `NotImplementedError`
— that's expected until filled in):

- **`src/models/unet.py`** — `VanillaUNet.forward()`: the skip-connection
  wiring for the from-scratch U-Net. Test: `tests/test_unet.py`.
- **`src/training/metrics.py`** — `dice_coefficient()` / `iou_score()`:
  the overlap-metric math. Test: `tests/test_metrics.py`.

Everything else (data loading, splitting, training loop, checkpointing,
losses, SAM2 harness) is fully implemented.

## Structure

```
configs/segmentation.yaml       # encoder, paths, hyperparams, device
src/data/dataset.py             # Dataset/DataLoader, RGB+mask pairs, augmentation
src/data/splits.py              # pig-disjoint, weight-stratified split logic
src/models/unet.py              # encoder-swappable U-Net (forward is a TODO)
src/training/losses.py          # Dice+BCE combined loss
src/training/metrics.py         # Dice/IoU eval metrics (math is a TODO)
src/training/train.py           # epoch loop, checkpointing, logging, CLI
src/evaluation/sam2_baseline.py # zero-shot SAM2 inference harness
tests/                          # pytest suite, including the two stub tests
```
