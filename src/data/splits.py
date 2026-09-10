"""Pig-disjoint, weight-stratified train/val/test splitting.

Implements the locked split design (see CLAUDE.md "Locked design decisions"):
group by (subset, pig_id) to prevent identity leakage across the same
physical animal's longitudinal re-weighings, stratify by weight quintile
computed from each group's mean weight, split 70/15/15 with a fixed seed.
Computed once, persisted, and reused by every downstream consumer
(segmentation, feature extraction, regression).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "mask_path",
    "subset",
    "pig_id",
    "weight_kg",
    "fold_dir",
    "has_mask",
}


def _validate_manifest(manifest: pd.DataFrame) -> None:
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")


def compute_group_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Collapse the manifest to one row per (subset, pig_id) group.

    Returns a DataFrame indexed by nothing in particular, with columns
    ["subset", "pig_id", "mean_weight_kg", "n_images", "n_masked_images"],
    one row per leakage-prevention group.
    """
    _validate_manifest(manifest)
    grouped = manifest.groupby(["subset", "pig_id"], as_index=False).agg(
        mean_weight_kg=("weight_kg", "mean"),
        n_images=("image_path", "size"),
        n_masked_images=("has_mask", "sum"),
    )
    return grouped


def assign_weight_quintiles(group_table: pd.DataFrame, n_quintiles: int = 5) -> pd.Series:
    """Assign each group a quintile bin (0..n_quintiles-1) from mean_weight_kg.

    Uses pandas.qcut with duplicate-edge handling so that ties in weight
    don't crash the binning on this modest-sized dataset (~74 groups).
    """
    try:
        bins = pd.qcut(group_table["mean_weight_kg"], q=n_quintiles, labels=False, duplicates="drop")
    except ValueError as exc:  # too few unique values for the requested quintile count
        raise ValueError(
            f"Could not form {n_quintiles} weight quintiles from "
            f"{group_table['mean_weight_kg'].nunique()} unique group weights."
        ) from exc
    return bins


def split_groups(
    manifest: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """Compute the pig-disjoint, weight-stratified 70/15/15 split.

    Returns a DataFrame with columns ["subset", "pig_id", "split"], where
    "split" is one of "train"/"val"/"test". Groups (not images) are the
    unit of assignment: every image belonging to a given (subset, pig_id)
    lands in the same split.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    group_table = compute_group_table(manifest)
    group_table = group_table.copy()
    group_table["quintile"] = assign_weight_quintiles(group_table, n_quintiles)

    rng = np.random.default_rng(seed)
    assignments = []

    for _, quintile_groups in group_table.groupby("quintile"):
        # Shuffle within quintile for a random but seeded, stratified split.
        shuffled = quintile_groups.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
        n = len(shuffled)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        # remainder goes to test, guarding against rounding drift
        n_test = n - n_train - n_val

        labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        shuffled = shuffled.assign(split=labels)
        assignments.append(shuffled)

    result = pd.concat(assignments, ignore_index=True)[["subset", "pig_id", "split"]]
    return result


def check_masked_subset_coverage(
    manifest: pd.DataFrame, split_table: pd.DataFrame, min_groups_per_split: int = 3
) -> None:
    """Assert the masked subset (rgb3394) has enough pig-groups per split.

    Only 42 pig-groups exist in the masked subset total, so a naive
    unified split can starve segmentation's val/test folds. Raises
    AssertionError with counts if any split falls below the threshold.
    """
    masked_pig_ids = set(
        manifest.loc[manifest["subset"] == "rgb3394", "pig_id"].unique()
    )
    masked_splits = split_table[
        (split_table["subset"] == "rgb3394") & (split_table["pig_id"].isin(masked_pig_ids))
    ]
    counts = masked_splits["split"].value_counts().to_dict()
    for split_name in ("train", "val", "test"):
        n = counts.get(split_name, 0)
        assert n >= min_groups_per_split, (
            f"Masked-subset (rgb3394) '{split_name}' split has only {n} pig-groups "
            f"(< {min_groups_per_split}); segmentation eval on this split would be "
            f"unreliable. Full masked-subset split counts: {counts}"
        )


def build_and_save_splits(
    manifest_path: str,
    output_path: str,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """Load a manifest CSV, compute the split, validate coverage, and save it.

    This is the single entry point that should be run once to produce
    data/splits.csv; every other module reads that file rather than
    recomputing the split.
    """
    manifest = pd.read_csv(manifest_path)
    split_table = split_groups(
        manifest,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        n_quintiles=n_quintiles,
    )
    check_masked_subset_coverage(manifest, split_table)
    split_table.to_csv(output_path, index=False)
    return split_table


def load_manifest_with_splits(manifest_path: str, splits_path: str) -> pd.DataFrame:
    """Join a manifest CSV to a previously-computed splits CSV.

    Returns the manifest with an added "split" column. Raises if any
    (subset, pig_id) group in the manifest has no split assignment
    (e.g. splits.csv is stale relative to the manifest).
    """
    manifest = pd.read_csv(manifest_path)
    split_table = pd.read_csv(splits_path)
    merged = manifest.merge(split_table, on=["subset", "pig_id"], how="left")
    unassigned = merged["split"].isna()
    if unassigned.any():
        bad_groups = merged.loc[unassigned, ["subset", "pig_id"]].drop_duplicates()
        raise ValueError(
            f"{len(bad_groups)} (subset, pig_id) groups in the manifest have no "
            f"split assignment in {splits_path} — is it stale? Groups: "
            f"{bad_groups.to_dict('records')}"
        )
    return merged
