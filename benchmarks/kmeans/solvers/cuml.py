from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    import cuml
    import cupy
    import numpy as np


class Solver(BaseSolver):
    """Note: not sure this solver actually should fit here.

    It is not documented wether it runs the lloyd algorithm or equivalent brute-force
    algorithm, and if the iterations are meant to result in the same inertia than
    other solvers on the bench ?
    """

    name = "cuml"
    requirements = ["cuml"]

    parameters = dict(device=["gpu"])

    stopping_criterion = SingleRunCriterion(1)

    def skip(self, **objective_dict):

        init = objective_dict["init"]
        if not hasattr(init, "copy") and (init == "k-means++"):
            return True, (
                "Support for k-means++ is not implemented in cuml. cuml only "
                "implements k-means|| whose walltime can't be compared with "
                "k-means++. "
            )

        algorithm = objective_dict["algorithm"]
        if algorithm != "lloyd":
            return True, "cuml only support the lloyd algorithm."

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
        sample_weight,
        init,
        n_clusters,
        n_init,
        max_iter,
        tol,
        verbose,
        algorithm,
        random_state,
    ):
        if self.device == "cpu":
            # Copy the data before running the benchmark to ensure that no unfortunate
            # side effects can happen
            self.X = X.copy()
            if hasattr(sample_weight, "copy"):
                sample_weight = sample_weight.copy()
            if hasattr(init, "copy"):
                init = init.copy()

        else:
            self.X = cupy.asarray(X)
            if hasattr(sample_weight, "copy"):
                sample_weight = cupy.asarray(sample_weight)
            if hasattr(init, "copy"):
                init = cupy.asarray(init)

        self.sample_weight = sample_weight
        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter

        if tol == 0:
            tol = 1e-16
        self.tol = tol

        self.verbose = verbose
        self.algorithm = algorithm
        self.random_state = random_state

    def warm_up(self):
        cuml.KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=1,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
        ).fit(self.X, sample_weight=self.sample_weight)

    def run(self, _):
        estimator = cuml.KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
        ).fit(self.X, sample_weight=self.sample_weight)
        self.inertia_ = estimator.inertia_
        self.n_iter_ = estimator.n_iter_

    def get_result(self):
        return dict(inertia=self.inertia_, n_iter=self.n_iter_, **self._parameters)
