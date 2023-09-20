from datetime import datetime

from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "Lloyd walltime"
    url = "https://github.com/soda-inria/sklearn-engine-benchmarks"

    requirements = ["numpy"]

    min_benchopt_version = "1.4.0"

    # Since our goal is to measure walltime for solvers that perform exact same
    # computations, the solver parameters are part of the objective and must be set
    # for all solvers, rather than being an independent benchmark space for each
    # solver.
    parameters = dict(
        n_clusters=[127],
        init=["random", "k-means++"],
        n_init=[1],
        max_iter=[100],
        tol=[0],
        verbose=[0],  # NB: for kmeans, verbosity can affect the performance
        algorithm=["lloyd"],
        random_state=[123],
        sample_weight=["None", "random"],
    )

    def set_data(self, X, **dataset_parameters):
        self.X = X
        dtype = X.dtype

        if self.init == "random" or self.sample_weight == "random":
            rng = np.random.default_rng(self.random_state)

        if self.sample_weight == "None":
            sample_weight = None
        elif self.sample_weight == "unary":
            sample_weight = np.ones(len(X), dtype=dtype)
        elif self.sample_weight == "random":
            sample_weight = rng.random(size=len(X)).astype(dtype)
        else:
            raise ValueError(
                "Expected 'sample_weight' parameter to be either equal to 'None', "
                f"'unary' or 'random', but got {sample_weight}."
            )

        if self.init == "random":
            init = np.array(
                rng.choice(X, self.n_clusters, replace=False), dtype=X.dtype
            )
        elif self.init == "k-means++":
            init = self.init
        else:
            raise ValueError(
                "Expected 'init' parameter to be either equal to 'random' or "
                f"'k-means++' but got {init}."
            )

        self.init_ = init
        self.sample_weight_ = sample_weight
        self.dataset_parameters = dataset_parameters

    def evaluate_result(self, inertia, n_iter, **solver_parameters):
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
            value=inertia,
            n_iter=n_iter,
            objective_param___name=self.name,
            **all_parameters,
        )

    def get_one_result(self):
        return dict(inertia=1, n_iter=100)

    def get_objective(self):
        return dict(
            X=self.X,
            sample_weight=self.sample_weight_,
            init=self.init_,
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            algorithm=self.algorithm,
            random_state=self.random_state,
        )
