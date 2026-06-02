import numpy as np
import pandas as pd


def scumble(label_imbalance_ratios):
    if len(label_imbalance_ratios) == 0:
        return None
    values = np.asarray(label_imbalance_ratios, dtype=float)
    # Compute the geometric mean in log space; ``values.prod()`` overflows to
    # ``inf`` for samples with many labels. Imbalance ratios are always >= 1,
    # so the logarithm is well defined.
    geometric_mean_ir = np.exp(np.mean(np.log(values)))
    arithmetic_mean_ir = values.mean()
    scumble_score = 1 - geometric_mean_ir / arithmetic_mean_ir
    return scumble_score

def get_irlbl(data: pd.DataFrame) -> tuple[int, pd.Series]:
    labels = data.columns
    label_frequencies = data[labels].sum()
    max_freq = label_frequencies.max()
    irlbl = max_freq / label_frequencies
    return max_freq, irlbl


def remedial_resample(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a dataset with REMEDIAL.

    Each high-SCUMBLE sample whose positive labels span both the majority and
    minority label groups is split into two copies: a majority copy (minority
    labels masked with ``NaN``) and a minority copy (majority labels masked
    with ``NaN``). All other samples are kept unchanged.

    Args:
        data (pd.DataFrame): Each column corresponds to a label, each row to an
            instance. The index stores the instance IDs. Only supply this
            function with train data, and only with label columns that are
            positive for at least one instance (all-zero columns yield an
            infinite imbalance ratio).

    Returns:
        pd.DataFrame: The resampled dataset. The index keeps the original
            instance IDs; a split sample appears twice under the same ID (once
            per copy), so callers must assign new unique IDs.
    """
    print("Resampling with REMEDIAL...")

    active_cols = data.columns[data.any(axis=0)]
    all_cols = data.columns
    if len(active_cols) < len(all_cols):
        print(
            f"Warning: {len(all_cols) - len(active_cols)} all-zero columns with undefined imbalance ratio will be dropped for resampling; they will be re-added as all-zero after resampling."
        )
    data = data[active_cols]

    _, irlbl = get_irlbl(data)
    meanir = irlbl.mean()
    print(f"Mean imbalance ratio: {meanir:.2f}")

    scumble_scores = data.apply(lambda row: scumble(irlbl[row == 1]), axis=1)
    scumble_mean = scumble_scores.mean()
    print(f"Mean scumble score: {scumble_mean}")

    # split labels into majority labels (irlbl <= meanir) and minority labels (irlbl > meanir)
    minority_labels = irlbl[irlbl > meanir].index
    majority_labels = irlbl[irlbl <= meanir].index

    # Rows with no positive labels get a NaN scumble score; they cannot be
    # split, so keep them unchanged rather than dropping them.
    nan_scumble_idx = data.index[scumble_scores.isna()]
    print(f"Number of rows with NaN scumble score (kept unchanged): {len(nan_scumble_idx)}")

    # Split only high-scumble rows whose positive labels span both label
    # groups. Rows with labels from just one side stay unchanged.
    candidate_rows = data[scumble_scores > scumble_mean]

    split_indices = []
    majority_rows = []
    minority_rows = []
    only_minority_rows = 0
    only_majority_rows = 0

    for _, row in candidate_rows.iterrows():
        has_majority = bool(row[majority_labels].any())
        has_minority = bool(row[minority_labels].any())
        if not (has_majority and has_minority):
            if has_majority:
                only_majority_rows += 1
            elif has_minority:
                only_minority_rows += 1
            continue

        split_indices.append(row.name)

        majority_row = row[data.columns].astype(float)
        majority_row.loc[minority_labels] = float("nan")
        majority_rows.append(majority_row)

        minority_row = row[data.columns].astype(float)
        minority_row.loc[majority_labels] = float("nan")
        minority_rows.append(minority_row)

    majority_rows = pd.DataFrame(majority_rows, columns=data.columns)
    minority_rows = pd.DataFrame(minority_rows, columns=data.columns)

    print(
        f"Number of majority rows to add: {len(majority_rows)}, number of minority rows to add: {len(minority_rows)}, number of original rows split: {len(split_indices)}"
    )
    print(
        f"Number of rows with only majority labels: {only_majority_rows}, number of rows with only minority labels: {only_minority_rows}"
    )

    # Drop only the rows that were actually split; everything else (including
    # NaN-scumble and single-group rows) is carried over unchanged. The index
    # (instance IDs) is preserved so callers can recover features per row.
    resampled_data = pd.concat(
        [
            data.drop(index=pd.Index(split_indices)),
            majority_rows,
            minority_rows,
        ],
    )

    print(
        "Data resampling completed, dataset size after resampling:",
        len(resampled_data),
    )
    # re-add all-0 columns as all-0, and cast to float32 (NaN masks must be preserved)
    resampled_data = resampled_data.astype(np.float32).reindex(columns=all_cols, fill_value=0.0)
    return resampled_data


