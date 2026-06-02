import random

import pandas as pd

from overbagging.datasets.chebi import ChebiDataset
from overbagging.oversampling import oversample
from overbagging.remedial import get_irlbl


def _imbalanced_dataset():
    """A dataset with one very frequent and two rare labels.

    ``maj`` is positive in every row, while ``min1``/``min2`` appear in a single
    row each, giving them a high imbalance ratio (they are the minority labels
    ML-ROS should oversample). The index uses string ids to mirror the real
    ChEBI binding (where the index carries instance idents).
    """
    rows = {
        "s0": [1, 0, 0],
        "s1": [1, 0, 0],
        "s2": [1, 0, 0],
        "s3": [1, 0, 0],
        "s4": [1, 1, 0],
        "s5": [1, 0, 1],
    }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["maj", "min1", "min2"]
    ).astype(bool)


def test_oversample_adds_samples_until_bags_fall_below_mean():
    data = _imbalanced_dataset()
    random.seed(0)
    result = oversample(data, sampling_rate=0.5)
    # The sampling budget (0.5 * 6 = 3) is not fully spent: each of the two
    # minority bags is duplicated once, which already drops its imbalance ratio
    # below the mean, so both bags leave the pool and the loop stops at 2 adds.
    assert len(result) == len(data) + 2


def test_oversample_only_duplicates_minority_samples():
    data = _imbalanced_dataset()
    random.seed(0)
    result = oversample(data, sampling_rate=0.5)

    added = result.iloc[len(data) :]
    # Every duplicated row must carry at least one minority label.
    assert (added[["min1", "min2"]].any(axis=1)).all()


def test_oversample_preserves_parent_ident_index():
    data = _imbalanced_dataset()
    random.seed(0)
    result = oversample(data, sampling_rate=0.5)

    # Duplicated rows keep their parent index label rather than being reindexed.
    assert set(result.index).issubset(set(data.index))
    added = result.iloc[len(data) :]
    assert set(added.index).issubset({"s4", "s5"})


def test_oversample_does_not_mutate_input():
    data = _imbalanced_dataset()
    before = data.copy(deep=True)
    random.seed(0)
    oversample(data, sampling_rate=0.5)
    pd.testing.assert_frame_equal(data, before)


def test_oversample_noop_when_sampling_rate_zero():
    data = _imbalanced_dataset()
    result = oversample(data, sampling_rate=0.0)
    pd.testing.assert_frame_equal(result, data)


def test_oversample_lowers_mean_imbalance_ratio():
    data = _imbalanced_dataset()
    random.seed(0)
    _, irlbl_before = get_irlbl(data)
    result = oversample(data, sampling_rate=1.0)
    _, irlbl_after = get_irlbl(result)
    assert irlbl_after.mean() < irlbl_before.mean()


# --------------------------------------------------------------------------- #
# ChebiDataset binding
# --------------------------------------------------------------------------- #
def test_chebi_oversample_labels_keeps_columns_and_parent_idents():
    data = _imbalanced_dataset()
    random.seed(0)
    resampled = ChebiDataset._oversample_labels(data, sampling_rate=0.5)

    # Same label columns, more (or equal) rows, every row maps back to a parent.
    assert list(resampled.columns) == list(data.columns)
    assert len(resampled) >= len(data)
    assert set(resampled.index).issubset(set(data.index))
