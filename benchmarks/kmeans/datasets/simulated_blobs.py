import numpy as np
from benchopt import BaseDataset
from benchopt.datasets import make_correlated_data


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        "n_samples, n_features": [(50_000_000, 14), (10_000_000, 14)],
        "dtype": ["float32", "float64"],
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

        return dict(X=X.astype(getattr(np, self.dtype)))
