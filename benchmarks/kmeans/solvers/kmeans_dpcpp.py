from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    import dpctl
    import dpctl.tensor as dpt
    import kmeans_dpcpp as kdp
    import numpy as np
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.exceptions import NotSupportedByEngineError
    from sklearn_numba_dpex.kmeans.engine import KMeansEngine

    class KMeansDPCPPEngine(KMeansEngine):
        def init_centroids(self, X, sample_weight):
            init = self.init
            if isinstance(init, str) and init == "k-means++":
                raise NotSupportedByEngineError(
                    "kmeans_dpcpp does not support k-means++ init."
                )
            return super().init_centroids(X, sample_weight)

        def kmeans_single(self, X, sample_weight, centers_init_t):
            n_samples, n_features = X.shape
            device = X.device.sycl_device

            assignments_idx = dpt.empty(n_samples, dtype=dpt.int32, device=device)
            res_centroids_t = dpt.empty_like(centers_init_t)

            # TODO: make work_group_size and centroid window dimension consistent in
            # kmeans_dpcpp with up to date configuration in sklearn_numba_dpex ?
            centroids_window_height = 8
            work_group_size = 128

            centroids_private_copies_max_cache_occupancy = 0.7

            n_iters, total_inertia = kdp.kmeans_lloyd_driver(
                X.T,
                sample_weight,
                centers_init_t,
                assignments_idx,
                res_centroids_t,
                self.tol,
                bool(self.estimator.verbose),
                self.estimator.max_iter,
                centroids_window_height,
                work_group_size,
                centroids_private_copies_max_cache_occupancy,
                X.sycl_queue,
            )

            return assignments_idx, total_inertia, res_centroids_t, n_iters


class Solver(BaseSolver):
    name = "kmeans-dpcpp"

    # NB: This requirement depends on `numba_dpex`. The user is expected to ensure that
    # it is installed properly along with the low level-runtime environment compatible
    # with the hardware when this solver runs. If a runtime is not available, the
    # corresponding benchmark will be skipped.
    # Moreover `kmeans_dpcpp` is expected to be installed from source following
    # special instructions in the README.
    requirements = [
        "git+https://github.com/soda-inria/sklearn-numba-dpex.git"
        "@168da1f8c751d4d33eed7c4880f3f734ac1edf0b#egg=sklearn-numba-dpex",
        "kmeans_dpcpp",
    ]

    parameters = dict(device=["cpu", "gpu"], runtime=["level_zero", "opencl"])

    stopping_criterion = SingleRunCriterion(1)

    def skip(self, **objective_dict):
        try:
            device = dpctl.SyclDevice(f"{self.runtime}:{self.device}")
        except Exception:
            return True, f"{self.runtime} runtime not found for device {self.device}"

        X = objective_dict["X"]
        if (X.dtype == np.float64) and not device.has_aspect_fp64:
            return True, (
                f"This {self.device} device has no support for float64 compute"
            )

        init = objective_dict["init"]
        if not hasattr(init, "copy") and (init == "k-means++"):
            return True, ("Support for k-means++ is not implemented in kmeans_dpcpp.")

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
        device = device = dpctl.SyclDevice(f"{self.runtime}:{self.device}")
        # Copy the data before running the benchmark to ensure that no unfortunate side
        # effects can happen
        self.X = dpt.asarray(X, copy=True, device=device)

        if hasattr(sample_weight, "copy"):
            sample_weight = dpt.asarray(sample_weight, copy=True, device=device)

        self.sample_weight = sample_weight

        if hasattr(init, "copy"):
            init = dpt.asarray(init, copy=True, device=device)

        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.algorithm = algorithm
        self.random_state = random_state

    def warm_up(self):
        with sklearn.config_context(engine_provider=KMeansDPCPPEngine):
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
        with sklearn.config_context(engine_provider=KMeansDPCPPEngine):
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
            self.inertia_ = estimator.inertia_.item()
            self.n_iter_ = estimator.n_iter_

    def get_result(self):
        return dict(
            inertia=self.inertia_,
            n_iter=self.n_iter_,
            version_info="dev",
            __name=self.name,
            **self._parameters,
        )
