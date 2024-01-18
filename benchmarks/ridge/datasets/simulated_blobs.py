from benchopt import BaseDataset, safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "Simulated_correlated_data"

    parameters = {
        "n_samples, n_features": [
            (10_000_000, 100),
            (5_000_000, 100),
            (5_000_000, 10),
            (5000, 10000),
            (5000, 5000),
        ],
        "n_targets": [1, 10],
        "dtype": ["float32"],
        "random_state": [123],
    }

    def __init__(self, n_samples, n_features, n_targets, dtype, random_state):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_targets = n_targets
        self.random_state = random_state
        self.dtype = dtype

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, self.n_targets, random_state=rng
        )

        dtype = getattr(np, self.dtype)

        return dict(
            X=X.astype(dtype), y=y.astype(dtype), __name=self.name, **self._parameters
        )
