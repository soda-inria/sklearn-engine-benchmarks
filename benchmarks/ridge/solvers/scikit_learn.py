from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    from sklearn.linear_model import Ridge
    from sklearn.linear_model._base import _rescale_data


class Solver(BaseSolver):
    name = "scikit-learn"
    requirements = ["scikit-learn"]

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
        self.X = X.copy()
        self.y = y.copy()

        if hasattr(sample_weight, "copy"):
            sample_weight = sample_weight.copy()
        self.sample_weight = sample_weight

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def run(self, _):
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

        self.coef_ = estimator.coef_

    def get_result(self):
        if self.sample_weight is not None:
            X, y, _ = _rescale_data(self.X, self.y, self.sample_weight, inplace=True)

        y = y.reshape((y.shape[0], -1))

        coef_ = self.coef_.reshape((X.shape[0], 1, -1))

        objective = ((y - (X @ coef_).squeeze(1)) ** 2).sum() + (
            len(y.T) * self.alpha * (coef_**2).sum()
        )

        return dict(
            objective=objective,
            version_info=f"scikit-learn {version('scikit-learn')}",
            __name=self.name,
            **self._parameters,
        )
