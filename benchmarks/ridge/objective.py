from datetime import datetime

from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model._base import _rescale_data


class Objective(BaseObjective):
    name = "Ridge walltime"
    url = "https://github.com/soda-inria/sklearn-engine-benchmarks"

    requirements = ["numpy"]

    # Since our goal is to measure walltime for solvers that perform exact same
    # computations, the solver parameters are part of the objective and must be set
    # for all solvers, rather than being an independent benchmark space for each
    # solver.
    parameters = {
        "alpha": [1.0],
        "fit_intercept": [True],
        "solver, max_iter, tol": [("svd", None, 0)],
        "sample_weight": ["None"],  # NB: add "random" to test non None weights
        "random_state": [123],
    }

    def set_data(self, X, y, **dataset_parameters):
        self.X = X
        self.y = y
        dtype = X.dtype

        if self.sample_weight == "None":
            sample_weight = None
        elif self.sample_weight == "unary":
            sample_weight = np.ones(len(X), dtype=dtype)
        elif self.sample_weight == "random":
            rng_sample_weight = np.random.default_rng(
                dataset_parameters["random_state"] + 1
            )
            sample_weight = rng_sample_weight.random(size=len(X)).astype(dtype)
        else:
            raise ValueError(
                "Expected 'sample_weight' parameter to be either equal to 'None', "
                f"'unary' or 'random', but got {sample_weight}."
            )

        self.sample_weight_ = sample_weight
        self.dataset_parameters = dataset_parameters

    def evaluate_result(self, weights, intercept, **solver_parameters):
        # NB: weights, intercept expected to be numpy arrays

        X, y = self.X, self.y
        if self.sample_weight_ is not None:
            X, y, _ = _rescale_data(X, y, self.sample_weight_, inplace=False)

        y = y.reshape((y.shape[0], -1))
        weights = weights.reshape((-1, X.shape[1], 1))

        value = (
            (((X @ weights).squeeze(2) + (intercept - y).T) ** 2).sum()
            + (self.alpha * (weights**2).sum())
        ) / (X.shape[0] * len(y.T))

        all_parameters = dict(solver_param_run_date=datetime.today())
        all_parameters.update(
            {
                ("dataset_param_" + key): value
                for key, value in self.dataset_parameters.items()
            }
        )
        all_parameters.update(
            {
                ("objective_param_" + key): value
                for key, value in self._parameters.items()
            }
        )
        all_parameters.update(
            {("solver_param_" + key): value for key, value in solver_parameters.items()}
        )
        return dict(
            value=value,
            objective_param___name=self.name,
            **all_parameters,
        )

    def get_one_result(self):
        n_features = self.dataset_parameters["n_features"]
        n_targets = self.dataset_parameters["n_targets"]
        if n_targets == 1:
            weights = np.ones((n_features,))
        else:
            weights = np.ones(
                (
                    n_targets,
                    n_features,
                )
            )

        return dict(weights=weights, intercept=np.ones((n_targets,)))

    def get_objective(self):
        # Copy the data before sending to the solver, to ensure that no unfortunate
        # side effects can happen
        X = self.X.copy()
        y = self.y.copy()

        sample_weight = self.sample_weight_
        if hasattr(sample_weight, "copy"):
            sample_weight = sample_weight.copy()

        return dict(
            X=X,
            y=y,
            sample_weight=sample_weight,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
