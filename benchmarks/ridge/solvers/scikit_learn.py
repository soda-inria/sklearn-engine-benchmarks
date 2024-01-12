from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    from sklearn.linear_model import Ridge


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
        self.X = X
        self.y = y
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

        self.weights = estimator.coef_
        self.intercept = estimator.intercept_

    def get_result(self):
        return dict(
            weights=self.weights,
            intercept=self.intercept,
            version_info=f"scikit-learn {version('scikit-learn')}",
            __name=self.name,
            **self._parameters,
        )
