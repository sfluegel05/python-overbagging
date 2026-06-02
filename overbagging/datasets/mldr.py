from pathlib import Path

import pandas as pd

from overbagging.datasets.base import BaseDataset

# Paper name -> "internal mldr name" used as the CSV filename stem (see
# DATASETS.md). The paper refers to e.g. ``yeast`` while the files on disk are
# named ``MultiLabelData_*``. Either spelling is accepted by ``from_name``.
PAPER_NAME_TO_STEM = {
    "cal500": "CAL500",
    "chess": "chess",
    "corel16k": "Corel16k001",
    "corel5k": "Corel5k",
    "delicious": "delicious_train",
    "enron": "enron",
    "mediamill": "mediamill-train-exp1",
    "medical": "MEDC",
    "tmc2007": "tmc2007-500-train",
    "yeast": "MultiLabelData",
}


class MldrDataset(BaseDataset):
    """Pipeline bindings for the mldr.datasets CSV exports used in REMEDIAL-HwR.

    Each dataset is a directory of CSVs (see ``DATASETS.md``). The pipeline
    needs two of them:

    * ``{stem}_labels.csv`` -- an ``ID`` column followed by one ``0/1`` column
      per label, one row per instance. ``ID`` becomes the DataFrame index and
      is the instance id. Keeping that id on every (possibly duplicated or
      split) output row lets a caller recover the matching feature row from
      ``{stem}_features.csv`` afterwards and assemble the modified dataset.
    * ``{stem}_folds.csv`` -- an ``ID`` column and a ``fold`` column assigning
      each instance to a cross-validation fold.

    One fold is designated the *holdout*: it is kept as-is (pass-through, never
    resampled) while the remaining folds form the train portion that the
    pipeline resamples.
    """

    def __init__(self, labels_csv_path, fold, fold_csv_path):
        self.labels_csv_path = labels_csv_path
        self.fold = fold
        self.fold_csv_path = fold_csv_path

    # ------------------------------------------------------------------ #
    # BaseDataset machinery.
    # ------------------------------------------------------------------ #
    def _load(self):
        """Read the binary label matrix, indexed by the ``ID`` column."""
        return pd.read_csv(self.labels_csv_path, index_col="ID")

    def _split(self, data):
        """Hold out ``self.fold``; the remaining folds are the train portion."""
        folds = pd.read_csv(self.fold_csv_path, index_col="ID")["fold"]
        folds = [folds.loc[idx] for idx in data.index]  # align folds with label matrix
        if self.fold not in set(folds):
            raise ValueError(
                f"Fold {self.fold!r} not found in {self.fold_csv_path}. "
                f"Available folds: {sorted(folds.dropna().unique())}."
            )

        train_data = data[[fold != self.fold for fold in folds]]
        passthrough_data = data[[fold == self.fold for fold in folds]]
        print(
            f"train: {len(train_data)}, holdout (fold {self.fold}): "
            f"{len(passthrough_data)}"
        )
        return train_data, passthrough_data

    def _build_label_df(self, train_data):
        """The loaded matrix already is the label DataFrame (id index, one
        column per label)."""
        return train_data

    def _reassemble(self, resampled_labels, train_data, passthrough_data):
        """Turn the resampled label table into an output frame.
        Keeps the index (possibly with duplicates) and labels
        """
        frames = [resampled_labels]
        if len(passthrough_data):
            frames.append(passthrough_data)
        combined = pd.concat(frames)
        # index should be named "ID"
        combined.index.name = "ID"
        return combined

    def _save(self, data, target_path):
        """Write the resampled label table (``id``, labels)."""
        data.to_csv(target_path, index=True)
