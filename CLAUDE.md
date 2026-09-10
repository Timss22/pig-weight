# pig-weight

Non-contact pig weight estimation from 2D RGB imagery (PIGRGB-Weight dataset).
Phase 1 of a multi-phase research pipeline: Segmentation → Feature extraction →
Regression. See `2026_Lab_Research_Proposal.pdf` (repo root's parent directory)
for full research context.

## Dataset ground truth

Findings from directly inspecting `data/PigRGB-Weight/` (Sept 2026) — record
here because they contradict assumptions a literature read alone would produce,
and re-deriving them cost real exploration time once already.

- **`RGB_9579/`** (folds 1-5): 9,578 images, 73 folders named `{weight_kg}_{id}`,
  but only **34 unique numeric `id` values**. The same `id` recurs at
  *different weights and in different folds* — e.g. id 103 appears as 160.4kg
  in fold3 and 192.48kg in fold2; id 106 appears at 112kg, 165kg, and 167kg
  across folds 2-4. **This is longitudinal data: the same physical pig
  reweighed over time as it grows**, not one pig per folder. The numeric `id`
  suffix is the true `pig_id` for leakage-prevention purposes — the folder
  name (`{weight}_{id}`) is a per-session label, not a pig identity.
- **`RGB_MASK_3394/`** (`RGB_3394` + `MASK_3394`, matched 1:1 by relative path,
  0 orphans either direction): 3,394 images, **42 unique `pig_id` values**,
  of which only **2 overlap** with the 34 ids in `RGB_9579`. Mean weight is
  77kg vs. 122kg in `RGB_9579` — this looks like a largely separate
  data-collection cohort, not a masked slice of the same animals. The 2
  coincidentally-shared ids are treated as **distinct groups per subset**
  (keyed on `(subset, pig_id)`, not `pig_id` alone) since the weight ranges
  don't support them being the same physical animal — see the Splits section
  below.
- Both subsets are 960×540 RGB PNGs. Masks in `MASK_3394` are binary
  (pixel values 0 / 255 only), stored as 3-channel RGB.
- These per-subset pig/id counts (34 and 42) don't match published counts for
  this dataset (e.g. one cited paper reports 124 pigs) — the published
  numbers likely refer to a different release or superset. Don't assume the
  literature's pig count applies to what's actually on disk here.
- The original `PIGRGB-Weight.zip` (9.5GB, redundant with the extracted
  `PigRGB-Weight/`) has been deleted after verifying the extraction was
  clean, to reclaim disk space on this machine (only ~54-58GB free on `/`).

## Locked design decisions

These were decided in project planning and should be implemented as-is, not
re-derived:

- **Splits**: pig-disjoint, grouped by `(subset, pig_id)` (see Dataset ground
  truth above for why the compound key). Stratified by weight quintile,
  computed from each group's **mean weight across its sessions/images**.
  One 70/15/15 split, fixed seed, computed once by `src/data/splits.py` from
  the `audit_pigrgb.py`-produced manifest, and reused by every downstream
  consumer — segmentation, feature extraction, and regression all read the
  same split assignment. Segmentation train/val/test is the subset of that
  split where `has_mask=True` (i.e. rows from `RGB_MASK_3394`).
- **Mask-handling policy**: ground-truth masks (`RGB_MASK_3394`) supervise
  and evaluate the segmentation model ONLY (Dice/IoU). They never reach
  feature extraction or anything downstream of segmentation — every image,
  including the ones with GT masks, uses the segmentation model's own
  *predicted* mask past the segmentation stage.
- **Manifest schema** (produced by `audit_pigrgb.py`, consumed by
  `src/data/splits.py` and `src/data/dataset.py`):
  `image_path, mask_path (nullable), subset (rgb9579|rgb3394), pig_id (int),
  weight_kg (float), fold_dir (nullable, fold1-5 for rgb9579 only), has_mask (bool)`
- **U-Net**: encoder-swappable, config-driven. Vanilla/from-scratch encoder is
  the default; swapping to pretrained `timm-mobilenetv3_small_100` or
  `resnet34` must be a config-only change, no code changes
  (segmentation-models-pytorch-style pattern).
- **SAM2 baseline scope**: zero-shot only (`sam2.1_hiera_base_plus`), used
  purely as a comparison point against the trained U-Net's Dice/IoU. The
  literature's "SAM2-Pig" reference (Applied Sciences 16(11), 5708, 2025) is
  actually a trained hybrid SAM2+ViT dorsal-segmentation network, not a
  zero-shot baseline — deliberately NOT matching that setup, since doing so
  would expand Phase 1 scope into later-phase territory. Keep SAM2 zero-shot.

## Compute environment

- Device selection is config-driven (`cuda` / `mps` / `cpu` / `auto`) — never
  hardcode a CUDA-only assumption.
- This Mac (Apple M2) is **dev/scaffolding only**: no NVIDIA GPU, MPS backend
  available via PyTorch, ~54-58GB free disk. Not intended for real training
  runs.
- Actual training machine is **TBD** — the research proposal names rented
  Vast.ai GPU instances; the user will confirm and add details here once
  settled.
