from pathlib import Path

import numpy as np
import pandas as pd

from overbagging.datasets.base import BaseDataset
from overbagging.remedial import get_irlbl, scumble


class ChebiDataset(BaseDataset):
    """Preprocessing pipeline bindings for a ChEBI ``data.pt`` dataset.

    A ChEBI ``data.pt`` is a list of dicts with keys ``ident`` (str), ``labels``
    (bool/float ndarray, one entry per label), ``features`` and ``group``.

    The generic machinery -- loading, splitting into the train portion (which is
    resampled) and the pass-through portion (validation/test, kept unchanged),
    and reassembling a resampled label table back into instance dicts -- is
    shared. Each pipeline step (``apply_remedial``, ``apply_bagging``, ...) only
    implements its own label-level transform.
    """

    def __init__(self, data_pt_path, splits_csv_path):
        self.data_pt_path = data_pt_path
        self.splits_csv_path = splits_csv_path

    def _load(self):
        import torch

        return torch.load(self.data_pt_path, weights_only=False)

    def _split(self, data):
        """Partition instances into train (resampled) and pass-through (val/test).

        Instances whose ``ident`` is absent from the split file are dropped.
        """
        split_dict = self._load_split_dict(self.splits_csv_path)
        train_data, passthrough_data, dropped = [], [], 0
        for instance in data:
            split = split_dict.get(instance["ident"])
            if split is None:
                dropped += 1
            elif split == "train":
                train_data.append(instance)
            else:
                passthrough_data.append(instance)
        print(
            f"train: {len(train_data)}, val/test: {len(passthrough_data)}, "
            f"dropped (not in splits): {dropped}"
        )
        return train_data, passthrough_data

    @staticmethod
    def _build_label_df(train_data):
        """Build the train label DataFrame: one row per instance (indexed by
        ``ident``), one column per label."""
        n_labels = len(train_data[0]["labels"])
        label_matrix = np.stack([inst["labels"] for inst in train_data])
        idents = [inst["ident"] for inst in train_data]
        return pd.DataFrame(label_matrix, index=idents, columns=range(n_labels))

    @staticmethod
    def _reassemble(resampled_labels, train_data, passthrough_data):
        """Rebuild instance dicts from a resampled label table + pass-through data.

        ``resampled_labels`` is indexed by parent instance ``ident`` (ids may
        repeat when a sample was duplicated or split); ``features`` and
        ``group`` are taken from the parent instance and the label vector is
        cast to ``float32`` (masked entries stay ``NaN``). Pass-through
        instances are appended unchanged, also cast to ``float32``.
        """
        features_by_ident = {inst["ident"]: inst for inst in train_data}
        values = resampled_labels.to_numpy(dtype=np.float32)

        new_data = []
        for ident, label_vec in zip(resampled_labels.index, values, strict=True):
            parent = features_by_ident[ident]
            new_data.append(
                {
                    "features": parent["features"],
                    "labels": label_vec,
                    "ident": ident,
                    "group": parent.get("group"),
                }
            )
        for inst in passthrough_data:
            new_data.append(
                {
                    "features": inst["features"],
                    "labels": np.asarray(inst["labels"]).astype(np.float32),
                    "ident": inst["ident"],
                    "group": inst.get("group"),
                }
            )
        return new_data
    
    def _save(self, data, target_path):
        import torch

        torch.save(data, target_path)

    @staticmethod
    def _load_split_dict(splits_csv_path):
        """Return a ``{instance_id: split_name}`` mapping from an ``id,split`` csv."""
        with open(splits_csv_path) as f:
            lines = f.read().splitlines()
        return {row[0]: row[1] for row in (line.split(",") for line in lines[1:])}

    # ------------------------------------------------------------------ #
    # Metrics.
    # ------------------------------------------------------------------ #
    @staticmethod
    def print_dataset_metrics(data):
        """Print basic metrics for a ChEBI-style dataset.

        Args:
            data: either a path to a ``.pt`` file or an already-loaded list of
                instance dicts (each with at least the keys ``ident`` and
                ``labels``).

        Returns:
            dict: the computed metrics (also printed).

        Mean IRLbl and mean SCUMBLE are computed over the labels that are
        positive for at least one instance; all-zero labels are excluded because
        their imbalance ratio is undefined (division by zero). Masked (``NaN``)
        labels are treated as not-positive.
        """
        if isinstance(data, str | Path):
            import torch

            data_name = Path(data).name
            data = torch.load(data, weights_only=False)
        else:
            data_name = "dataset"
        n_rows = len(data)
        n_ids = len({inst["ident"] for inst in data})

        label_matrix = np.stack(
            [np.asarray(inst["labels"], dtype=float) for inst in data]
        )
        label_df = pd.DataFrame(label_matrix)

        # Restrict to labels positive somewhere (freq > 0); NaN is skipped by sum.
        active_cols = label_df.columns[label_df.sum() > 0]
        active_df = label_df[active_cols]

        max_freq, irlbl = get_irlbl(active_df)
        mean_irlbl = irlbl.mean()

        scumble_scores = active_df.apply(lambda row: scumble(irlbl[row == 1]), axis=1)
        mean_scumble = scumble_scores.mean()

        print(f"Metrics for {data_name}:")
        print(f"  number of rows:        {n_rows}")
        print(f"  number of unique ids:  {n_ids}")
        print(f"  number of labels:      {len(active_cols)} (positive in >=1 row)")
        print(f"  mean IRLbl score:      {mean_irlbl:.4f}")
        print(f"  mean SCUMBLE score:    {mean_scumble:.4f}")
        print(f"  max frequency:         {max_freq:.4f}")

        return {
            "n_rows": n_rows,
            "n_ids": n_ids,
            "n_labels": len(active_cols),
            "mean_irlbl": float(mean_irlbl),
            "mean_scumble": float(mean_scumble),
            "max_freq": float(max_freq),
        }
