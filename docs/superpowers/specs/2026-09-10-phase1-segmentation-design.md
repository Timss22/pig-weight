# Phase 1: Segmentation — Design Spec

Date: 2026-09-10
Status: Approved, moving to implementation

## Context

First hands-on phase of the pig-weight research pipeline (non-contact pig
weight estimation from 2D RGB imagery, PIGRGB-Weight dataset). Full pipeline
is segmentation → morphometric feature extraction → regression; this spec
covers segmentation only. See `CLAUDE.md` for dataset ground truth and the
locked design decisions this spec implements.

## Goal

A trained, encoder-swappable U-Net that isolates the pig/dorsal region from
the pen, evaluated with Dice/IoU against `RGB_MASK_3394`'s ground-truth
masks, plus a zero-shot SAM2 comparison baseline. Two components (U-Net
forward pass, Dice/IoU metric math) are intentionally left as stubs with
tests, for the user to implement as a learning exercise — everything else
is production-quality.

## Manifest contract

Produced by `audit_pigrgb.py` (user-authored, outside this spec's scope),
consumed by `src/data/splits.py` and `src/data/dataset.py`:

| column | type | notes |
|---|---|---|
| `image_path` | str | relative to `data/PigRGB-Weight/` |
| `mask_path` | str, nullable | null for `rgb9579` rows |
| `subset` | `rgb9579` \| `rgb3394` | |
| `pig_id` | int | numeric folder suffix; see CLAUDE.md for longitudinal semantics |
| `weight_kg` | float | |
| `fold_dir` | str, nullable | `fold1`-`fold5`, `rgb9579` only |
| `has_mask` | bool | true only for `rgb3394` rows |

## Split logic (`src/data/splits.py`)

1. Group rows by `(subset, pig_id)` — the leakage-prevention unit. The 2
   numerically-coincident ids shared between subsets are kept as distinct
   groups (see CLAUDE.md: weight ranges don't support them being the same
   animal).
2. For each group, compute mean `weight_kg` across its rows/sessions.
3. Stratify groups into weight quintiles (5 bins) using those group-level
   means.
4. Within each quintile, split groups 70/15/15 into train/val/test with a
   fixed seed (`42`, configurable).
5. Persist the resulting `(subset, pig_id) -> split` assignment to
   `data/splits.csv` (or config-driven path) so it's computed once and
   reused by every consumer.
6. Assert: the masked subset's val/test folds each contain at least a
   handful of pig-groups (not near-zero) — only 42 groups exist there
   total, so this is a real risk to catch early, not a hypothetical.

Segmentation train/val/test = manifest rows where `has_mask=True`, joined
to this split assignment. Later phases (feature extraction, regression)
use the full manifest joined to the same assignment.

## Mask-handling policy

Ground-truth masks supervise/evaluate segmentation only (Dice/IoU against
`RGB_MASK_3394`). Every image — including ones with GT masks — uses the
trained U-Net's own *predicted* mask for everything past segmentation.

## Directory structure

```
configs/segmentation.yaml       # encoder choice, paths, hyperparams, device
src/data/dataset.py             # Dataset/DataLoader, RGB+mask pairs, augmentations
src/data/splits.py              # split logic above
src/models/unet.py              # STUB: forward()/decoder — TODO(timotius)
src/training/losses.py          # Dice+BCE combined loss (full impl)
src/training/metrics.py         # STUB: Dice/IoU math — TODO(timotius)
src/training/train.py           # epoch loop, checkpointing, logging, args
src/evaluation/sam2_baseline.py # zero-shot SAM2 inference harness
tests/test_unet.py              # shape-check tests against the stub
tests/test_metrics.py           # known-case tests (perfect/zero overlap, etc.)
requirements.txt
README.md
```

## Component design

**`src/data/dataset.py`** — `PigSegmentationDataset(manifest_df, split, transform)`
reads `has_mask=True` rows for the given split, loads RGB + binary mask
pairs (960×540, mask values {0,255} → binarized to {0,1}), applies
augmentation. Augmentation uses a **generous color-jitter range**
(brightness/contrast/saturation/hue) per the eventual lighting-robustness
need for open-house deployment — plus standard geometric augmentation
(flips, rotation, scale) appropriate for top-down imagery. Built on
`albumentations` for joint image+mask transforms.

**`src/models/unet.py`** — `PigUNet(nn.Module)`, config-driven encoder
(`vanilla` custom encoder by default; `timm-mobilenetv3_small_100` or
`resnet34` via `segmentation_models_pytorch` when configured — encoder
swap is config-only, no code changes). Constructor, config handling, and
full docstrings (expected input/output shapes: `(B,3,H,W) -> (B,1,H,W)`
logits) are implemented. `forward()` and decoder block wiring are
`# TODO(timotius): implement`.

**`src/training/losses.py`** — `DiceBCELoss` combining `nn.BCEWithLogitsLoss`
and a differentiable soft Dice loss, weighted sum (configurable weights).
Full implementation.

**`src/training/metrics.py`** — `dice_coefficient(pred, target, threshold)`
and `iou_score(pred, target, threshold)`, full signatures and docstrings
(binary masks, `pred` as logits or probabilities, threshold-binarized).
Math body is `# TODO(timotius): implement`.

**`src/training/train.py`** — epoch loop, `DiceBCELoss`, Adam/AdamW
optimizer, LR scheduling, checkpointing (best-val-Dice + last), logging
(stdout + CSV; a `--wandb` flag is out of scope unless requested),
CLI args (config path, resume, device override). Device resolution:
`config.device` of `cuda`/`mps`/`cpu`/`auto` (auto picks in that priority
order via `torch.cuda.is_available()` / `torch.backends.mps.is_available()`).

**`src/evaluation/sam2_baseline.py`** — loads `sam2.1_hiera_base_plus`,
runs zero-shot inference (automatic mask generation, or box/point prompt
derived from image-center heuristic given single-pig top-down framing —
implementation detail decided during coding, documented in the module
docstring), evaluates Dice/IoU against the same `rgb3394` test split as
the U-Net for a fair comparison. No training — zero-shot only, per locked
scope (see CLAUDE.md on why the literature's SAM2-Pig setup is NOT matched).

## Testing

`tests/test_unet.py`: instantiate `PigUNet` with default config, assert
`forward()` (once implemented by the user) produces `(B,1,H,W)` logits for
a range of input sizes/batch sizes; assert encoder-swap config produces a
model that still runs forward with correct output shape.

`tests/test_metrics.py`: known-input cases — perfect overlap (Dice/IoU = 1),
zero overlap (Dice/IoU = 0), partial overlap (hand-computed expected
value), empty prediction vs. non-empty target (edge case), batch-of-N
correctness (per-sample vs. mean-reduction modes if both are exposed).

Both test files are written now, against the stub signatures; they will
fail (or error) until the user fills in the TODOs — that's intended.

## Out of scope for this spec

Feature extraction, regression, SAM2 training/fine-tuning, on-device
deployment, edge hardware benchmarking — later phases per the research
proposal.
