"""Multilabel imbalance metrics (REMEDIAL-HwR, Charte et al. 2019, Table 1).

Unlike the original ``compute_metrics.py`` -- which resolves a dataset *name* to
a fixed ``datasets/<internal>/csv`` layout -- these helpers take CSV *file
paths* directly, so they work on any label/feature file: the raw mldr exports,
a single fold, or the resampled output of the pipeline.

Metrics:
  n        number of instances
  d        number of features (only if a features file is supplied)
  q        number of labels
  LCard    label cardinality  = mean positive labels per instance
  LDens    label density      = LCard / q
  DL       distinct labelsets = number of unique label combinations
  DL%      DL as a percentage of n
  MeanIR   mean imbalance ratio (mean of the finite per-label IRLbl)
  SCUMBLE  score of concurrence among imbalanced labels

Per-label: IRLbl(j) and its SCUMBLE contribution SCUMBLE_lbl(j).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from overbagging.remedial import get_irlbl, scumble

# Columns that the pipeline carries alongside the 0/1 label columns; they are
# not labels (or features) and must be excluded from the metric computation.
NON_LABEL_COLS = ("ID", "id", "parent_id")


def _drop_non_label_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop id/parent_id bookkeeping columns, leaving only label/feature columns."""
    drop = [c for c in NON_LABEL_COLS if c in df.columns]
    return df.drop(columns=drop) if drop else df


def load_label_matrix(labels_csv_path) -> pd.DataFrame:
    """Read a labels CSV into a pure label matrix (one column per label).

    Any ``ID``/``parent_id`` columns are dropped. Masked labels (``NaN``, as
    produced by REMEDIAL) are preserved.
    """
    return _drop_non_label_cols(pd.read_csv(labels_csv_path))


def count_features(features_csv_path) -> int:
    """Number of feature columns in a features CSV (excluding the id column)."""
    return _drop_non_label_cols(pd.read_csv(features_csv_path)).shape[1]


def _per_sample_scumble(labels: pd.DataFrame, irlbl: pd.Series) -> np.ndarray:
    """SCUMBLE contribution of each instance, positionally (index-duplicate safe).

    Instances with fewer than two active labels contribute 0. Returns an array
    aligned to ``labels`` row order, so duplicate indices (from oversampling)
    are handled correctly.
    """
    ir_values = irlbl.to_numpy()
    matrix = labels.to_numpy()
    scores = np.zeros(len(matrix))
    for i, row in enumerate(matrix):
        active = np.where(row == 1)[0]
        s = scumble(ir_values[active])
        scores[i] = 0.0 if s is None else s
    return scores


def compute_all_metrics(labels: pd.DataFrame, n_features: int | None = None) -> dict:
    """Compute every REMEDIAL-HwR metric for a label matrix.

    Args:
        labels: instances x labels 0/1 matrix (``NaN`` masks tolerated).
        n_features: feature count for the ``d`` metric; ``None`` if unknown.
    """
    n, q = labels.shape
    if n == 0 or q == 0:
        raise ValueError("Label matrix has no instances or no labels.")
    if labels.isna().to_numpy().any():
        print(
            "Warning: label matrix contains NaN (masked) entries; they are "
            "treated as inactive for the metrics."
        )

    _, irlbl = get_irlbl(labels)
    finite_ir = irlbl[np.isfinite(irlbl)]
    mean_ir = float(finite_ir.mean()) if len(finite_ir) else 0.0

    sample_scumble = _per_sample_scumble(labels, irlbl)
    overall_scumble = float(sample_scumble.mean())

    # Per-label SCUMBLE = mean instance-scumble over instances where the label
    # is active (only instances with >= 2 active labels contribute).
    matrix = labels.to_numpy()
    per_label_sum = np.zeros(q)
    per_label_n = np.zeros(q, dtype=int)
    for i, row in enumerate(matrix):
        active = np.where(row == 1)[0]
        if len(active) < 2:
            continue
        per_label_sum[active] += sample_scumble[i]
        per_label_n[active] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        scumble_lbl = np.where(per_label_n > 0, per_label_sum / per_label_n, 0.0)

    lcard = float(labels.sum(axis=1).mean())
    dl = len(set(map(tuple, labels.fillna(0).astype(int).to_numpy().tolist())))

    return {
        "n": n,
        "d": n_features,
        "q": q,
        "LCard": lcard,
        "LDens": lcard / q,
        "DL": dl,
        "DL%": 100.0 * dl / n,
        "MeanIR": mean_ir,
        "SCUMBLE": overall_scumble,
        "IRLbl": irlbl.to_numpy(),
        "SCUMBLE_lbl": scumble_lbl,
        "label_names": list(labels.columns),
    }


def print_dataset_metrics(
    labels_csv_path,
    features_csv_path=None,
    name: str | None = None,
    per_label: bool = True,
) -> dict:
    """Print the REMEDIAL-HwR metrics for a labels CSV (and optional features CSV).

    Args:
        labels_csv_path: path to a ``*_labels.csv`` (raw, a fold, or pipeline
            output -- ``ID``/``parent_id`` columns are ignored).
        features_csv_path: optional path to the matching features CSV; supplied
            only to report the feature count ``d``.
        name: label for the printout; defaults to the labels file stem.
        per_label: also print the per-label IRLbl / SCUMBLE_lbl breakdown.

    Returns:
        The metrics dict from :func:`compute_all_metrics`.
    """
    if name is None:
        name = Path(labels_csv_path).stem
    labels = load_label_matrix(labels_csv_path)
    n_features = count_features(features_csv_path) if features_csv_path else None
    m = compute_all_metrics(labels, n_features=n_features)

    d_str = "n/a" if m["d"] is None else f"{m['d']}"
    print(f"\n=== Dataset metrics: {name} ===")
    print(f"  n (instances)        : {m['n']}")
    print(f"  d (features)         : {d_str}")
    print(f"  q (labels)           : {m['q']}")
    print(f"  LCard (cardinality)  : {m['LCard']:.4f}")
    print(f"  LDens (density)      : {m['LDens']:.4f}")
    print(f"  DL (distinct sets)   : {m['DL']} ({m['DL%']:.1f}% of n)")
    print(f"  MeanIR               : {m['MeanIR']:.4f}")
    print(f"  SCUMBLE              : {m['SCUMBLE']:.5f}")

    if per_label:
        print(f"\n  {'Label':<30} {'IRLbl':>10} {'SCUMBLE_lbl':>12}")
        print(f"  {'-' * 54}")
        for lbl, ir, sc in zip(m["label_names"], m["IRLbl"], m["SCUMBLE_lbl"]):
            print(f"  {str(lbl):<30} {ir:>10.4f} {sc:>12.6f}")
    return m
