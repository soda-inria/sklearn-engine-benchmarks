from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    import cuml
    import cupy
    import numpy as np


class Solver(BaseSolver):
    """https://docs.rapids.ai/api/cuml/stable/api/#ridge-regression"""

    name = "cuml"
    requirements = ["cuml"]

    parameters = dict(device=["gpu"])

    stopping_criterion = SingleRunCriterion(1)

    def skip(self, **objective_dict):

        X = objective_dict["X"]
        if X.dtype == np.float64:
            # We haven't came accross cuda devices that doesn't support float64 yet,
            # can it happen ? If it happens, the following instruction will fail,
            # please enclose it with the appropriate Try/Except to return the
            # appropriate skip decision.
            cupy.zeros(1, dtype=cupy.float64)
            # return True, (
            #     f"This {self.device} device has no support for float64 compute"
            # )

        y = objective_dict["y"]
        if (y.ndim == 2) and (y.shape[1] > 1):
            return True, "Multitarget is not supported."

        solver = objective_dict["solver"]
        if solver != "svd":
            return True, "Only accepts the svd solver at the moment."

        return False, None

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
        self.X = cupy.asarray(X)
        self.y = cupy.asarray(y)

        self.sample_weight = sample_weight
        if sample_weight is not None:
            self.sample_weight = cupy.asarray(sample_weight)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def warm_up(self):
        sample_weight = self.sample_weight
        if sample_weight is not None:
            sample_weight = sample_weight[:2]
        cuml.Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
        ).fit(self.X[:2], self.y[:2], sample_weight=sample_weight)

    def run(self, _):
        estimator = cuml.Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
        ).fit(self.X, self.y, sample_weight=self.sample_weight)

        self.weights = estimator.coef_
        self.intercept = estimator.intercept_

    def get_result(self):
        return dict(
            weights=cupy.asnumpy(self.weights),
            intercept=cupy.asnumpy(self.intercept),
            n_iter=None,
            version_info=f"scikit-learn {version('scikit-learn')}",
            __name=self.name,
            **self._parameters,
        )
