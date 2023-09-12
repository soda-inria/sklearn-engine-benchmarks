from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    # isort: off
    import dpctl
    import numpy as np
    import sklearn
    from daal4py.sklearn.cluster._k_means_0_23 import (
        _daal4py_compute_starting_centroids,
        _daal4py_k_means_fit,
        getFPType,
        support_usm_ndarray,
    )
    from sklearn.cluster import KMeans
    from sklearn.cluster._kmeans import KMeansCythonEngine
    from sklearn.exceptions import NotSupportedByEngineError
    from sklearnex import config_context as sklearnex_config_context

    # isort: on

    class DAAL4PYEngine(KMeansCythonEngine):
        engine_name = "kmeans"

        def prepare_fit(self, X, y=None, sample_weight=None):
            if sample_weight is not None and any(sample_weight != sample_weight[0]):
                raise NotSupportedByEngineError(
                    "Non unary sample_weight is not supported by daal4py."
                )

            return super().prepare_fit(X, y, sample_weight)

        @support_usm_ndarray()
        def init_centroids(self, X, sample_weight):
            init = self.init
            _, centers_init = _daal4py_compute_starting_centroids(
                X,
                getFPType(X),
                self.estimator.n_clusters,
                init,
                self.estimator.verbose,
                self.random_state,
            )
            return centers_init

        @support_usm_ndarray()
        def kmeans_single(self, X, sample_weight, centers_init):
            cluster_centers, labels, inertia, n_iter = _daal4py_k_means_fit(
                X,
                nClusters=self.estimator.n_clusters,
                numIterations=self.estimator.max_iter,
                tol=self.tol,
                cluster_centers_0=centers_init,
                n_init=self.estimator.n_init,
                verbose=self.estimator.verbose,
                random_state=self.random_state,
            )

            return labels, inertia, cluster_centers, n_iter

        def get_labels(self, X, sample_weight):
            raise NotSupportedByEngineError

        def get_euclidean_distances(self, X):
            raise NotSupportedByEngineError

        def get_score(self, X, sample_weight):
            raise NotSupportedByEngineError


class Solver(BaseSolver):
    name = "scikit-learn-intelex"

    # NB: This requirement depends on `numba_dpex`. The user is expected to ensure that
    # it is installed properly along with the low level-runtime environment compatible
    # with the hardware when this solver runs. If a runtime is not available, the
    # corresponding benchmark will be skipped.
    requirements = [
        "git+https://github.com/soda-inria/sklearn-numba-dpex.git"
        "@168da1f8c751d4d33eed7c4880f3f734ac1edf0b#egg=sklearn-numba-dpex",
        "scikit-learn-intelex",
    ]

    parameters = dict(
        device=["cpu", "gpu"],
    )

    stopping_criterion = SingleRunCriterion(1)

    def skip(self, **objective_dict):
        try:
            device = dpctl.SyclDevice(f"{self.device}")
        except Exception:
            return True, f"{self.device} device not found."

        X = objective_dict["X"]
        if (X.dtype == np.float64) and not device.has_aspect_fp64:
            return True, (
                f"This {self.device} device has no support for float64 compute"
            )

        init = objective_dict["init"]
        if (init == "k-means++") and self.device == "gpu":
            return True, "gpu support for k-means++ is not implemented in daal4py."

        sample_weight = objective_dict["sample_weight"]
        if sample_weight is not None and any(sample_weight != sample_weight[0]):
            return True, "Non unary sample_weight is not supported by daal4py."

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
        with sklearnex_config_context(
            target_offload=self.device
        ), sklearn.config_context(engine_provider=DAAL4PYEngine):
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
        with sklearnex_config_context(
            target_offload=self.device
        ), sklearn.config_context(engine_provider=DAAL4PYEngine):
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
        print(self.inertia_)

    def get_result(self):
        return {"inertia": self.inertia_, "n_iter": self.n_iter_}
