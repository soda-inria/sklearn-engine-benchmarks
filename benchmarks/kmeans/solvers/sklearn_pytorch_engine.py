from importlib.metadata import PackageNotFoundError, version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    import sklearn
    import sklearn_pytorch_engine  # necessary to pre-load torch extensions  # noqa
    import torch
    from sklearn.cluster import KMeans
    from sklearn_pytorch_engine._utils import has_fp64_support


class Solver(BaseSolver):
    name = "sklearn-pytorch-engine"

    # NB: This requirement depends on torch. The user is expected to ensure that torch
    # is installed properly for the targeted backend ("cuda", "mps", "xpu", "hip",...)
    # when this solver runs. If a backend is not available, the corresponding
    # benchmark will be skipped.
    requirements = ["sklearn-pytorch-engine"]

    parameters = dict(device=["cpu", "xpu", "cuda", "mps"])
    stopping_criterion = SingleRunCriterion(1)

    def skip(self, **objective_dict):
        try:
            torch.zeros(1, dtype=torch.float32, device=self.device)
        except Exception:
            return True, f"{self.device} compute backend for pytorch not found"

        X = objective_dict["X"]
        if (X.dtype == np.float64) and not has_fp64_support(self.device):
            return True, (
                f"This {self.device} device has no support for float64 compute"
            )

        return False, None

    def set_objective(
        self,
        X,
        sample_weight,
        init,
        n_clusters,
        n_init,
        max_iter,
        tol,
        verbose,
        algorithm,
        random_state,
    ):
        device = self.device
        # Copy the data before running the benchmark to ensure that no unfortunate side
        # effects can happen
        self.X = torch.asarray(X, copy=True, device=self.device)

        if hasattr(sample_weight, "copy"):
            sample_weight = torch.asarray(sample_weight, copy=True, device=device)

        self.sample_weight = sample_weight

        if hasattr(init, "copy"):
            init = torch.asarray(init, copy=True, device=device)

        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.algorithm = algorithm
        self.random_state = random_state

    def warm_up(self):
        with sklearn.config_context(engine_provider="sklearn_pytorch_engine"):
            KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=1,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self.random_state,
                copy_x=False,
                algorithm=self.algorithm,
            ).fit(self.X, y=None, sample_weight=self.sample_weight)

    def run(self, _):
        with sklearn.config_context(engine_provider="sklearn_pytorch_engine"):
            estimator = KMeans(
                n_clusters=self.n_clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self.random_state,
                copy_x=False,
                algorithm=self.algorithm,
            ).fit(self.X, y=None, sample_weight=self.sample_weight)
            self.inertia_ = estimator.inertia_
            self.n_iter_ = estimator.n_iter_

    def get_result(self):
        version_info = f"sklearn-pytorch-engine dev; torch {version('torch')}"
        try:
            version_info += f"; ipex {version('intel-extension-for-pytorch')}"
        except PackageNotFoundError:
            pass

        return dict(
            inertia=self.inertia_,
            n_iter=self.n_iter_,
            version_info=version_info,
            __name=self.name,
            **self._parameters,
        )
