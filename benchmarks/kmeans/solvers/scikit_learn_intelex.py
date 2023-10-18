from contextlib import nullcontext
from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    # isort: off
    import dpctl
    import numpy as np
    from sklearnex.cluster import KMeans
    from sklearnex import config_context

    # isort: on


class Solver(BaseSolver):
    name = "scikit-learn-intelex"

    requirements = [
        "scikit-learn-intelex",
        "dpcpp-cpp-rt",
    ]

    parameters = {
        "device, runtime": [
            ("cpu", "numpy"),
            ("gpu", "level_zero"),
        ]
    }

    stopping_criterion = SingleRunCriterion(1)

    def skip(self, **objective_dict):
        if self.runtime != "numpy":
            try:
                device = dpctl.SyclDevice(f"{self.runtime}:{self.device}")
            except Exception:
                return (
                    True,
                    f"{self.runtime} runtime not found for device {self.device}",
                )

            X = objective_dict["X"]
            if (X.dtype == np.float64) and not device.has_aspect_fp64:
                return True, (
                    f"This {self.device} device has no support for float64 compute"
                )

        init = objective_dict["init"]
        if (
            (not hasattr(init, "copy"))
            and (init == "k-means++")
            and (self.device != "cpu")
        ):
            return True, (
                "support for k-means++ is not implemented in scikit-learn-intelex "
                "for devices other than cpu."
            )

        sample_weight = objective_dict["sample_weight"]
        if sample_weight is not None:
            return True, (
                "sample_weight != None is not supported by scikit-learn-intelex."
            )

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
        # TODO: the overhead of the copy of the data from host to device could be
        # eliminated if scikit-learn-intelex could just take usm_ndarray objects as
        # input and directly run compute with the underlying memory buffer. The
        # documentation at
        # https://intel.github.io/scikit-learn-intelex/latest/oneapi-gpu.html#device-offloading  # noqa
        # suggests that it is the intended behavior, however in practice
        # scikit-learn-intelex currently always perform underlying copies
        # under the hood no matter what, and sometimes fails at doing so. See e.g.
        # issue at
        # https://github.com/intel/scikit-learn-intelex/issues/1534#issuecomment-1766266299  # noqa

        # if self.runtime != "numpy":
        #     device = device = dpctl.SyclDevice(f"{self.runtime}:{self.device}")
        #     self.X = dpt.asarray(X, copy=True, device=device)

        #     if hasattr(sample_weight, "copy"):
        #         sample_weight = dpt.asarray(sample_weight, copy=True, device=device)

        #     if hasattr(init, "copy"):
        #         init = dpt.asarray(init, copy=True, device=device)
        # else:
        #     self.X = X.copy()
        #     if hasattr(sample_weight, "copy"):
        #         sample_weight = sample_weight.copy()
        #     if hasattr(init, "copy"):
        #         init = init.copy()

        # Copy the data before running the benchmark to ensure that no unfortunate
        # side effects can happen
        self.X = X.copy()
        if hasattr(sample_weight, "copy"):
            sample_weight = sample_weight.copy()
        if hasattr(init, "copy"):
            init = init.copy()

        self.sample_weight = sample_weight
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
        with nullcontext() if (self.runtime == "numpy") else config_context(
            target_offload=f"{self.runtime}:{self.device}"
        ):
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
        return dict(
            inertia=self.inertia_,
            n_iter=self.n_iter_,
            version_info=f"scikit-learn-intelex {version('scikit-learn-intelex')}",
            __name=self.name,
            **self._parameters,
        )
