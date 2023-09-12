from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    from sklearn.cluster import KMeans


class Solver(BaseSolver):
    name = "scikit-learn"
    requirements = ["scikit-learn"]

    stopping_criterion = SingleRunCriterion(1)

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
        # Copy the data before running the benchmark to ensure that no unfortunate side
        # effects can happen
        self.X = X.copy()
        if hasattr(sample_weight, "copy"):
            sample_weight = sample_weight.copy()
        self.sample_weight = sample_weight
        if hasattr(init, "copy"):
            init = init.copy()

        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.algorithm = algorithm
        self.random_state = random_state

    def warm_up(self):
        KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=1,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=False,
            algorithm=self.algorithm,
        ).fit(self.X, y=None, sample_weight=self.sample_weight)

    def run(self, _):
        estimator = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=False,
            algorithm=self.algorithm,
        ).fit(self.X, y=None, sample_weight=self.sample_weight)
        self.inertia_ = estimator.inertia_
        self.n_iter_ = estimator.n_iter_

    def get_result(self):
        return {"inertia": self.inertia_, "n_iter": self.n_iter_}
