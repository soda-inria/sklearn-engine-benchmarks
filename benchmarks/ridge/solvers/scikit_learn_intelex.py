from contextlib import nullcontext
from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    # isort: off
    import dpctl
    import numpy as np
    from sklearnex.linear_model import Ridge
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
    }

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
        #     self.X = X

        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

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

        solver = objective_dict["solver"]

        if solver != "DefaultDense":
            # TODO: investigate ?
            return True, "The only supported solver parameter is DefaultDense."

        return False, None

    def warm_up(self):
        n_warmup_samples = 20
        n_warmup_features = 5
        sample_weight = self.sample_weight
        if sample_weight is not None:
            sample_weight = sample_weight[:n_warmup_samples]
        with nullcontext() if (self.runtime is None) else config_context(
            target_offload=f"{self.runtime}:{self.device}"
        ):
            Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                copy_X=False,
                max_iter=self.max_iter,
                tol=self.tol,
                solver="auto",
                positive=True if (self.solver == "lbfgs") else False,
                random_state=self.random_state,
            ).fit(
                self.X[:n_warmup_samples, :n_warmup_features],
                self.y[:n_warmup_samples],
                sample_weight,
            )

    def run(self, _):
        with nullcontext() if (self.runtime is None) else config_context(
            target_offload=f"{self.runtime}:{self.device}"
        ):
            estimator = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                copy_X=False,
                max_iter=self.max_iter,
                tol=self.tol,
                solver="auto",
                positive=True if (self.solver == "lbfgs") else False,
                random_state=self.random_state,
            ).fit(self.X, self.y, self.sample_weight)

        self.weights = estimator.coef_
        self.intercept = estimator.intercept_
        self.n_iter_ = estimator.n_iter_

    def get_result(self):
        return dict(
            weights=self.weights,
            intercept=self.intercept,
            n_iter=self.n_iter_,
            version_info=f"scikit-learn-intelex {version('scikit-learn-intelex')}",
            __name=self.name,
            **self._parameters,
        )
