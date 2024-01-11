from datetime import datetime

from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


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
        "sample_weight": ["None", "random"],
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

    def evaluate_result(self, objective, **solver_parameters):
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
            value=objective,
            objective_param___name=self.name,
            **all_parameters,
        )

    def get_one_result(self):
        return dict(objective=1)

    def get_objective(self):
        return dict(
            X=self.X,
            y=self.y,
            sample_weight=self.sample_weight_,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
