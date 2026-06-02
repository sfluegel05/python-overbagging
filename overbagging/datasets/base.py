import abc

from overbagging.bagging import bootstrap_data
from overbagging.oversampling import oversample
from overbagging.remedial import remedial_resample


class BaseDataset(abc.ABC):
    # ------------------------------------------------------------------ #
    # Generic pipeline machinery.
    # ------------------------------------------------------------------ #
    def _run(self, label_transform, target_path):
        data = self._load()
        train_data, passthrough_data = self._split(data)
        label_df = self._build_label_df(train_data)
        resampled = label_transform(label_df)
        new_data = self._reassemble(resampled, train_data, passthrough_data)

        self._save(new_data, target_path)
        print(f"Saved {len(new_data)} instances to {target_path}")
        return new_data

    def _load(self):
        """Load the dataset from disk."""
        raise NotImplementedError

    def _split(self, data):
        """Partition instances into train (resampled) and pass-through (val/test)."""
        raise NotImplementedError

    def _build_label_df(self, train_data):
        """Build the train label DataFrame: one row per instance (indexed by
        ``ident``), one column per label."""
        raise NotImplementedError

    def _reassemble(self, resampled_labels, train_data, passthrough_data):
        """Reassemble the resampled train labels with the original train data and
        passthrough val/test data, yielding a new dataset to save."""
        raise NotImplementedError

    def _save(self, data, target_path):
        """Save the resampled dataset to disk."""
        raise NotImplementedError

        # ------------------------------------------------------------------ #

    # Pipeline steps -- the only step-specific code.
    # ------------------------------------------------------------------ #
    def apply_remedial(self, target_path, split_fraction: float | None = None):
        """Apply REMEDIAL resampling to the train split and save to ``target_path``.

        Train samples that REMEDIAL splits are masked: the masked labels become
        ``NaN`` in the resulting ``float32`` label vector. Split copies keep
        their parent instance's ``ident`` (they remain train samples, so the
        original split assignment still applies).

        ``split_fraction`` (in ``[0, 1]``) selects that fraction of the train
        samples -- the highest-SCUMBLE ones -- as split candidates; when
        ``None`` (the default), candidates are every sample above the mean
        SCUMBLE score. See :func:`~overbagging.remedial.remedial_resample`.
        """
        return self._run(
            lambda label_df: remedial_resample(label_df, split_fraction),
            target_path,
        )

    def apply_bagging(self, target_path, random_state: int = 42):
        """Build a bagging (bootstrap) resample of the train split and save it.

        Draws train instances with replacement (as many
        as there are train instances). Duplicated
        instances keep their parent ``ident``; validation/test instances pass
        through unchanged.
        """
        return self._run(
            lambda label_df: bootstrap_data(label_df, seed=random_state),
            target_path,
        )

    def apply_oversampling(self, target_path, sampling_rate=0.1):
        """Apply ML-ROS random oversampling to the train split and save it.

        Minority samples -- those carrying labels whose imbalance ratio exceeds
        the mean imbalance ratio -- are duplicated until roughly
        ``sampling_rate * n_train`` new samples have been added. Duplicated
        instances keep their parent ``ident``; validation/test instances pass
        through unchanged.
        """
        return self._run(
            lambda label_df: oversample(label_df, sampling_rate),
            target_path,
        )
