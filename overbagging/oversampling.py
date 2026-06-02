import random

import numpy as np
import pandas as pd

from overbagging.remedial import get_irlbl

def oversample(
    data: pd.DataFrame,
    sampling_rate: float = 0.1,
) -> pd.DataFrame:
    """
    Oversample the dataset using ML-ROS.

    Args:
        data (pd.DataFrame): The dataset. Columns are labels, rows are instances.
        sampling_rate (float): The rate at which to oversample the dataset.

    Returns:
        pd.DataFrame: The oversampled dataset. Duplicated rows keep the index
            label (instance id) of the sample they were copied from, so the
            returned index may contain repeated values.
    """
    # reset index to ensure it is unique
    old_index = list(data.index)
    data = data.reset_index(drop=True)
    samples_to_add = sampling_rate * len(data)
    print(f"Need to add {samples_to_add:.2f} samples to data")
    # calculate label imbalance ratios. Labels that are never positive in this
    # split have frequency 0 and therefore an infinite imbalance ratio; they
    # cannot be oversampled (no instance carries them) and would make MeanIR
    # infinite, so drop them before computing the mean and selecting minorities.
    max_freq, irlbl = get_irlbl(data)
    n_empty = int(np.isinf(irlbl).sum())
    if n_empty:
        print(f"Ignoring {n_empty} all-zero label(s) with undefined imbalance ratio")
        irlbl = irlbl[np.isfinite(irlbl)]
    meanir = irlbl.mean()
    print(f"Mean imbalance ratio: {meanir:.2f}")
    # get bags for all labels where irlbl > meanir
    minority_labels = irlbl[irlbl > meanir].index
    print(f"Oversampling {len(minority_labels)} minority labels")
    minority_bags = dict()
    for label in minority_labels:
        minority_bags[label] = list(data[data[label] == 1].index)
    new_samples = []
    round_idx = 1
    while samples_to_add > 0 and len(minority_bags) > 0:
        minority_bags_next_round = dict()
        for label, bag in minority_bags.items():
            new_sample = bag[random.randint(0, len(bag) - 1)]
            bag.append(new_sample)
            new_samples.append(new_sample)
            samples_to_add -= 1
            if samples_to_add <= 0:
                break
            irlbl_bag = max_freq / len(bag)
            if irlbl_bag > meanir:
                minority_bags_next_round[label] = bag
        minority_bags = minority_bags_next_round
        if round_idx % 5 == 0:
            print(
                f"Round {round_idx} finished, {samples_to_add:.2f} samples to go, {len(minority_bags)} minority bags left"
            )
        round_idx += 1

    # ``new_samples`` holds index labels (instance ids), so select with ``loc``;
    # keep the index intact so callers can map duplicates back to their parent.
    new_samples_df = data.loc[new_samples]
    print(f"Adding {len(new_samples_df)} samples to data")
    data = pd.concat([data, new_samples_df])
    # restore original index with instance ids
    data.index = [old_index[i] for i in data.index]

    return data