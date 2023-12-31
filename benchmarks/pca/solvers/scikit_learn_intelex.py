from contextlib import nullcontext
from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    # isort: off
    import dpctl
    import numpy as np
    from sklearnex.decomposition import PCA
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
            ("cpu", None),  # TODO: replace "None" with "opencl" if relevant
            ("gpu", "level_zero"),
        ],
        "svd_solver, power_iteration_normalizer": [
            ("full", None),
            ("randomized", "LU"),
            ("arpack", None),
        ],
        "iterated_power": ["auto"],
    }

    stopping_criterion = SingleRunCriterion(1)

    def skip(self, **objective_dict):
        if self.runtime is not None:
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

        return False, None

    def set_objective(
        self,
        X,
        n_components,
        tol,
        n_oversamples,
        random_state,
        verbose,
    ):
        if (
            self.svd_solver in {"full", "arpack"}
            and self.power_iteration_normalizer is not None
        ):
            raise ValueError(
                f"svd_solver {self.svd_solver} can only run if "
                "power_iteration_normalizer parameter is set to None, but got "
                f"power_iteration_normalizer={self.power_iteration_normalizer}"
            )

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
        # else:
        #     self.X = X.copy)()

        # Copy the data before running the benchmark to ensure that no unfortunate
        # side effects can happen
        self.X = X.copy()

        self.n_components = n_components
        self.tol = tol
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.verbose = verbose

    def run(self, _):
        power_iteration_normalizer = self.power_iteration_normalizer
        if power_iteration_normalizer is None:
            power_iteration_normalizer = "auto"

        with nullcontext() if (self.runtime is None) else config_context(
            target_offload=f"{self.runtime}:{self.device}"
        ):
            estimator = PCA(
                n_components=self.n_components,
                copy=True,
                whiten=False,
                svd_solver=self.svd_solver,
                tol=self.tol,
                iterated_power=self.iterated_power,
                n_oversamples=self.n_oversamples,
                power_iteration_normalizer=power_iteration_normalizer,
                random_state=self.random_state,
            ).fit(self.X, y=None)

        self.explained_variance_ratio_ = estimator.explained_variance_ratio_

    def get_result(self):
        return dict(
            explained_variance_ratio_sum=self.explained_variance_ratio_.sum(),
            version_info=f"scikit-learn-intelex {version('scikit-learn-intelex')}",
            __name=self.name,
            **self._parameters,
        )
