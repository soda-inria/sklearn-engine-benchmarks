from importlib.metadata import PackageNotFoundError, version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    # isort: off
    import numpy as np

    # NB: even if it's not use for the compute we rely on the sklearn_pytorch_engine
    # for the few utilities it contains e.g for loading torch with xpu support and
    # checking for float64 compat.
    # This import is necessary to pre-load torch extensions
    import sklearn_pytorch_engine  # noqa
    import torch
    from sklearn import config_context
    from sklearn.decomposition import PCA
    from sklearn_pytorch_engine._utils import has_fp64_support

    # isort: on


class Solver(BaseSolver):
    name = "sklearn-torch-dispatch"
    requirements = ["scikit-learn", "sklearn-pytorch-engine"]

    parameters = {
        "svd_solver, power_iteration_normalizer": [
            ("full", None),
            ("randomized", "QR"),
        ],
        "device": ["cpu", "xpu", "cuda", "mps"],
        "iterated_power": ["auto"],
    }

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
        n_components,
        tol,
        n_oversamples,
        random_state,
        verbose,
    ):
        if (
            self.svd_solver in {"full", "arpack"}
            and self.power_iteration_normalizer is not None
        ):
            raise ValueError(
                f"svd_solver {self.svd_solver} can only run if "
                "power_iteration_normalizer parameter is set to None, but got "
                f"power_iteration_normalizer={self.power_iteration_normalizer}"
            )

        # Copy the data before running the benchmark to ensure that no unfortunate side
        # effects can happen
        self.X = torch.asarray(X, copy=True, device=self.device)

        self.n_components = n_components
        self.tol = tol
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.verbose = verbose

    def run(self, _):
        power_iteration_normalizer = self.power_iteration_normalizer
        if power_iteration_normalizer is None:
            power_iteration_normalizer = "auto"

        with config_context(array_api_dispatch=True):
            estimator = PCA(
                n_components=self.n_components,
                copy=True,
                whiten=False,
                svd_solver=self.svd_solver,
                tol=self.tol,
                iterated_power=self.iterated_power,
                n_oversamples=self.n_oversamples,
                power_iteration_normalizer=power_iteration_normalizer,
                random_state=self.random_state,
            ).fit(self.X, y=None)

        self.explained_variance_ratio_ = estimator.explained_variance_ratio_

    def get_result(self):
        version_info = (
            f"scikit-learn {version('scikit-learn')}; torch {version('torch')}"
        )
        try:
            version_info += f"; ipex {version('intel-extension-for-pytorch')}"
        except PackageNotFoundError:
            pass

        return dict(
            explained_variance_ratio_sum=float(self.explained_variance_ratio_.sum()),
            version_info=version_info,
            __name=self.name,
            **self._parameters,
        )
