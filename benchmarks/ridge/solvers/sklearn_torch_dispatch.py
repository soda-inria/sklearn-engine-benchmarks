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
    from sklearn.linear_model import Ridge
    from sklearn_pytorch_engine._utils import has_fp64_support

    # isort: on


class Solver(BaseSolver):
    name = "sklearn-torch-dispatch"
    requirements = ["scikit-learn", "sklearn-pytorch-engine"]

    parameters = {
        "device": ["cpu", "xpu", "cuda", "mps"],
    }

    stopping_criterion = SingleRunCriterion(1)

    def set_objective(
        self,
        X,
        y,
        sample_weight,
        alpha,
        fit_intercept,
        solver,
        max_iter,
        tol,
        random_state,
    ):
        # Copy the data before running the benchmark to ensure that no unfortunate side
        # effects can happen
        self.X = torch.asarray(X, copy=True, device=self.device)
        self.y = torch.asarray(y, copy=True, device=self.device)
        self.sample_weight = sample_weight
        if sample_weight is not None:
            self.sample_weight = torch.asarray(
                sample_weight, copy=True, device=self.device
            )

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def skip(self, **objective_dict):
        if not Ridge()._get_tags()["array_api_support"]:
            return True, (
                "Requires the development branch for Ridge support for Array API."
            )

        try:
            torch.zeros(1, dtype=torch.float32, device=self.device)
        except Exception:
            return True, f"{self.device} compute backend for pytorch not found"

        X = objective_dict["X"]
        if (X.dtype == np.float64) and not has_fp64_support(self.device):
            return True, (
                f"This {self.device} device has no support for float64 compute"
            )

        solver = objective_dict["solver"]
        if solver != "svd":
            return True, "Only accepts the svd solver at the moment."

        return False, None

    def warm_up(self):
        n_warmup_samples = 20
        n_warmup_features = 5
        sample_weight = self.sample_weight
        if sample_weight is not None:
            sample_weight = sample_weight[:n_warmup_samples].copy()
        with config_context(array_api_dispatch=True):
            Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                copy_X=False,
                max_iter=self.max_iter,
                tol=self.tol,
                solver=self.solver,
                positive=True if (self.solver == "lbfgs") else False,
                random_state=self.random_state,
            ).fit(
                self.X[:n_warmup_samples, :n_warmup_features].copy(),
                self.y[:n_warmup_samples].copy(),
                sample_weight,
            )

    def run(self, _):
        with config_context(array_api_dispatch=True):
            estimator = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                copy_X=False,
                max_iter=self.max_iter,
                tol=self.tol,
                solver=self.solver,
                positive=True if (self.solver == "lbfgs") else False,
                random_state=self.random_state,
            ).fit(self.X, self.y, self.sample_weight)

        self.weights = estimator.coef_
        self.intercept = estimator.intercept_
        self.n_iter_ = estimator.n_iter_

    def get_result(self):
        version_info = (
            f"scikit-learn {version('scikit-learn')}; torch {version('torch')}"
        )
        try:
            version_info += f"; ipex {version('intel-extension-for-pytorch')}"
        except PackageNotFoundError:
            pass

        return dict(
            weights=self.weights.cpu().numpy(),
            intercept=self.intercept.cpu().numpy(),
            n_iter=self.n_iter_,
            version_info=version_info,
            __name=self.name,
            **self._parameters,
        )
