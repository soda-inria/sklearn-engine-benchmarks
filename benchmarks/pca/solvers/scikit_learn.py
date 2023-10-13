from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    from sklearn.decomposition import PCA


class Solver(BaseSolver):
    name = "scikit-learn"
    requirements = ["scikit-learn"]

    parameters = {
        "svd_solver, power_iteration_normalizer": [
            (svd_solver, power_iteration_normalizer)
            for svd_solver in ["full", "randomized"]
            for power_iteration_normalizer in ["LU"]
        ]
        + [("arpack", "none")],
        "iterated_power": ["auto"],
    }

    stopping_criterion = SingleRunCriterion(1)

    def set_objective(
        self,
        X,
        n_components,
        tol,
        iterated_power,
        n_oversamples,
        random_state,
        verbose,
    ):
        # Copy the data before running the benchmark to ensure that no unfortunate side
        # effects can happen
        self.X = X.copy()

        self.n_components = n_components
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.verbose = verbose

    def run(self, _):
        estimator = PCA(
            n_components=self.n_components,
            copy=True,
            whiten=False,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
            random_state=self.random_state,
        ).fit(self.X, y=None)

        self.explained_variance_ratio_ = estimator.explained_variance_ratio_

    def get_result(self):
        return dict(
            explained_variance_ratio_sum=self.explained_variance_ratio_.sum(),
            version_info=f"scikit-learn {version('scikit-learn')}",
            __name=self.name,
            **self._parameters,
        )
