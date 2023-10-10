from importlib.metadata import version

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion

with safe_import_context() as import_ctx:
    # isort: off
    import dpctl
    import dpctl.tensor as dpt
    import numpy as np
    from sklearnex.decomposition import PCA

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
            ("cpu", "level_zero"),
            ("cpu", "opencl"),
            ("gpu", "level_zero"),
            ("gpu", "opencl"),
        ],
        "svd_solver": ["full", "arpack", "randomized"],
        "power_iteration_normalizer": ["QR", "LU", "none"],
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

        svd_solver = objective_dict["svd_solver"]
        power_iteration_normalizer = objective_dict["power_iteration_normalizer"]

        if (svd_solver == "arpack") and power_iteration_normalizer != "none":
            return True, (
                "arpack solver expect power iteration normalizer parameter set to "
                "'none'"
            )

        return False, None

    def set_objective(
        self,
        X,
        n_components,
        whiten,
        tol,
        iterated_power,
        n_oversamples,
        random_state,
        verbose,
    ):

        # Copy the data before running the benchmark to ensure that no unfortunate
        # side effects can happen
        if self.runtime != "numpy":
            device = device = dpctl.SyclDevice(f"{self.runtime}:{self.device}")
            self.X = dpt.asarray(X, copy=True, device=device)

        else:
            self.X = X.copy()

        self.components = n_components
        self.whiten = whiten
        self.tol = tol
        self.iterated_power = self.iterated_power
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.verbose = verbose

    def run(self, _):
        estimator = PCA(
            n_components=self.n_components,
            copy=False,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            n_oversamples=self.n_oversamples,
            power_iteration_normalizer=self.power_iteration_normalizer,
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
