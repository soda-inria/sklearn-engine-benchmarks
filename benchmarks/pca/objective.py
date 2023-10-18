from datetime import datetime

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "PCA walltime"
    url = "https://github.com/soda-inria/sklearn-engine-benchmarks"

    requirements = ["numpy"]

    # Since our goal is to measure walltime for solvers that perform exact same
    # computations, the solver parameters are part of the objective and must be set
    # for all solvers, rather than being an independent benchmark space for each
    # solver.
    parameters = dict(
        n_components=[10],
        tol=[0.0],
        n_oversamples=[10],
        random_state=[123],
        verbose=[False],
    )

    def set_data(self, X, **dataset_parameters):
        self.X = X
        self.dataset_parameters = dataset_parameters

    def evaluate_result(self, explained_variance_ratio_sum, **solver_parameters):
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
            value=explained_variance_ratio_sum,
            objective_param___name=self.name,
            **all_parameters,
        )

    def get_one_result(self):
        return dict(explained_variance_ratio_sum=1)

    def get_objective(self):
        return dict(
            X=self.X,
            n_components=self.n_components,
            tol=self.tol,
            iterated_power=self.iterated_power,
            n_oversamples=self.n_oversamples,
            random_state=self.random_state,
            verbose=self.verbose,
        )
