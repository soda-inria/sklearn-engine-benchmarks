from benchopt import BaseDataset, safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "Simulated_correlated_data"

    parameters = {
        "n_samples, n_features": [
            (20_000_000, 100),
            (15_000, 15_000),
            (2_000_000, 100),
            (5000, 5000),
        ],
        "dtype": ["float32"],
        "random_state": [123],
    }

    def __init__(self, n_samples, n_features, dtype, random_state):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.dtype = dtype

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        X, *_ = make_correlated_data(self.n_samples, self.n_features, random_state=rng)

        return dict(
            X=X.astype(getattr(np, self.dtype)), __name=self.name, **self._parameters
        )
