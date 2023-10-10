from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    import cuml
    import cupy
    import numpy as np


class Solver(BaseSolver):
    name = "cuml"
    requirements = ["cuml"]

    parameters = dict(
        device=["gpu"],
        svd_solver=["full", "jacobi"],
    )

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

        return False, None

    def set_objective(
        self,
        X,
        n_components,
        whiten,
        tol,
        iterated_power,
        n_oversamples,
        random_state,
        verbose,
    ):
        if self.device == "cpu":
            # Copy the data before running the benchmark to ensure that no unfortunate
            # side effects can happen
            self.X = X.copy()

        else:
            self.X = cupy.asarray(X)

        self.components = n_components
        self.whiten = whiten
        self.tol = tol

        # if tol == 0:
        #     tol = 1e-16
        # self.tol = tol

        self.iterated_power = self.iterated_power
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.verbose = verbose

    def run(self, _):
        estimator = cuml.PCA(
            copy=False,
            iterated_power=self.iterated_power,
            n_components=self.n_components,
            random_state=self.random_state,
            svd_solver=self.svd_solver,
            tol=self.tol,
            whiten=self.whiten,
        ).fit(self.X, y=None)

        self.explained_variance_ratio_ = estimator.explained_variance_ratio_

    def get_result(self):
        return dict(
            explained_variance_ratio_sum=self.explained_variance_ratio_.sum().item(),
            version_info=f"cuml {version('cuml')}",
            __name=self.name,
            **self._parameters,
        )
