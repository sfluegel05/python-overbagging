import numpy as np
import pandas as pd
import pytest

from overbagging.remedial import get_irlbl, remedial_resample, scumble


# --------------------------------------------------------------------------- #
# get_irlbl
# --------------------------------------------------------------------------- #
def test_get_irlbl_ratios():
    # frequencies: a=4, b=2, c=1  ->  max_freq = 4
    data = pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "b": [1, 1, 0, 0],
            "c": [1, 0, 0, 0],
        }
    )
    max_freq, irlbl = get_irlbl(data)
    assert max_freq == pytest.approx(4.0)
    assert irlbl["a"] == pytest.approx(1.0)  # most frequent label -> ratio 1
    assert irlbl["b"] == pytest.approx(2.0)
    assert irlbl["c"] == pytest.approx(4.0)


def test_get_irlbl_is_one_when_all_labels_equally_frequent():
    data = pd.DataFrame({"a": [1, 0], "b": [1, 0]})
    max_freq, irlbl = get_irlbl(data)
    assert max_freq == pytest.approx(1.0)
    assert irlbl["a"] == pytest.approx(1.0)
    assert irlbl["b"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# scumble
# --------------------------------------------------------------------------- #
def test_scumble_empty_returns_none():
    assert scumble(pd.Series([], dtype=float)) is None


def test_scumble_single_value_is_zero():
    # geometric mean == arithmetic mean == the value itself
    assert scumble(pd.Series([5.0])) == pytest.approx(0.0)


def test_scumble_equal_values_is_zero():
    assert scumble(pd.Series([2.0, 2.0, 2.0])) == pytest.approx(0.0)


def test_scumble_divergent_values():
    # geom = sqrt(1*9) = 3, arith = 5  ->  1 - 3/5 = 0.4
    assert scumble(pd.Series([1.0, 9.0])) == pytest.approx(0.4)


def test_scumble_between_zero_and_one_for_imbalanced_input():
    score = scumble(pd.Series([1.0, 4.0, 100.0]))
    assert 0.0 < score < 1.0


# --------------------------------------------------------------------------- #
# remedial_resample
# --------------------------------------------------------------------------- #
def _crafted_dataset():
    """A small dataset with two clearly mixed (high-SCUMBLE) rows.

    Columns maj1/maj2 are frequent (majority group), min1/min2 are rare
    (minority group). Rows ``m1`` and ``m2`` each carry one majority and one
    minority label, so they have a high SCUMBLE score and span both groups.
    Every other row sits in a single group.
    """
    cols = ["maj1", "maj2", "min1", "min2"]
    rows = {
        "a0": [1, 1, 0, 0],
        "a1": [1, 1, 0, 0],
        "a2": [1, 1, 0, 0],
        "a3": [1, 1, 0, 0],
        "a4": [1, 1, 0, 0],
        "a5": [1, 1, 0, 0],
        "m1": [1, 0, 1, 0],  # spans both groups -> expected split
        "m2": [0, 1, 0, 1],  # spans both groups -> expected split
        "minonly": [0, 0, 1, 0],
        "majonly": [1, 0, 0, 0],
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols).astype(bool)


def test_remedial_resample_splits_only_mixed_high_scumble_rows():
    data = _crafted_dataset()
    resampled = remedial_resample(data)

    counts = resampled.index.value_counts()
    split_ids = sorted(counts[counts == 2].index)
    assert split_ids == ["m1", "m2"]
    # every non-split original row appears exactly once
    for ident in data.index.difference(split_ids):
        assert counts[ident] == 1
    # total rows = originals + one extra copy per split row
    assert len(resampled) == len(data) + len(split_ids)


def test_remedial_resample_masks_are_complementary_and_lossless():
    data = _crafted_dataset()
    resampled = remedial_resample(data)

    for ident in ("m1", "m2"):
        copies = resampled.loc[ident]
        assert len(copies) == 2
        c1 = copies.iloc[0].to_numpy(dtype=float)
        c2 = copies.iloc[1].to_numpy(dtype=float)

        nan1, nan2 = np.isnan(c1), np.isnan(c2)
        # each column is masked in exactly one of the two copies
        assert np.all(nan1 ^ nan2)
        assert not np.any(nan1 & nan2)

        # recombining the unmasked halves reconstructs the original row
        merged = np.where(nan1, c2, c1)
        original = data.loc[ident].to_numpy(dtype=float)
        np.testing.assert_array_equal(merged, original)


def test_remedial_resample_keeps_unsplit_rows_unchanged():
    data = _crafted_dataset()
    resampled = remedial_resample(data)

    for ident in ("a0", "minonly", "majonly"):
        row = resampled.loc[ident]
        np.testing.assert_array_equal(
            row.to_numpy(dtype=float), data.loc[ident].to_numpy(dtype=float)
        )


def test_remedial_resample_does_not_mutate_input():
    data = _crafted_dataset()
    before = data.copy(deep=True)
    remedial_resample(data)
    pd.testing.assert_frame_equal(data, before)
