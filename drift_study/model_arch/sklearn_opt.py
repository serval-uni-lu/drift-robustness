from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import optuna
from joblib import parallel_backend
from mlc.metrics.metric import Metric
from mlc.metrics.metrics import PredClassificationMetric
from mlc.models.model import Model
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
import time

class TimeOptimizer(Model):
    def __init__(
        self,
        model: Model,
        metric: Metric,
        n_trials: int = 25,
        n_splits: int = 5,
        **kwargs: Dict[str, Any],
    ):
        name = f"opt_{model.name}"
        super().__init__(
            name=name,
            objective=model.objective,
            **kwargs,
        )
        self.n_jobs = kwargs.get("n_jobs")
        self.model = model
        self.metric = metric
        self.n_trials = n_trials
        self.n_splits = n_splits

    def load(self, path: str) -> None:
        self.model.load(path)

    def save(self, path: str) -> None:
        self.model.save(path)

    @staticmethod
    def _compute_metric(
        model: Model,
        x: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        if model.objective in ["regression"]:
            return model.predict(x)
        if model.objective in ["binary", "classification"]:
            return model.predict_proba(x)

        raise NotImplementedError

    def _objective_one(
        self,
        model,
        x_train: npt.NDArray[np.float_],
        x_test: npt.NDArray[np.float_],
        y_train: npt.NDArray[np.float_],
        y_test: npt.NDArray[np.float_],
    ) -> float:
        start = time.time()
        # with parallel_backend("threading"):
        model.fit(x_train, y_train)
        y_scores = self._compute_metric(model, x_test)
        if isinstance(self.metric, PredClassificationMetric):
            y_scores = np.argmax(y_scores, axis=1)
        metric = self.metric.compute(y_test, y_scores)
        elasped = time.time() - start
        print(f"Train_time {elasped}")
        return metric

    def _objective(
        self,
        trial: Trial,
        trial_params: Dict[str, Any],
        x: npt.NDArray[np.float_],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
    ) -> float:
        model_params = self.model.define_trial_parameters(trial, trial_params)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        metrics = [
            self._objective_one(
                self.model.__class__(
                    n_jobs=self.n_jobs, verbose=1, **model_params
                ),
                x[train_index],
                x[test_index],
                y[train_index],
                y[test_index],
            )
            for (train_index, test_index) in reversed(list(tscv.split(x)))
        ]

        return float(np.mean(metrics))

    def fit(
        self,
        x: npt.NDArray[np.float_],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        x_val: Optional[npt.NDArray[np.float_]] = None,
        y_val: Optional[
            Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]
        ] = None,
    ) -> None:

        trial_params = {"n_features": x.shape[1]}

        if x_val is not None:
            raise NotImplementedError

        sampler = TPESampler(n_startup_trials=5, seed=42)
        study = optuna.create_study(
            study_name="train",
            sampler=sampler,
            directions=["maximize"],
            load_if_exists=True,
        )
        default_params = self.model.get_default_params(trial_params)
        if default_params is not None:
            study.enqueue_trial(default_params)

        study.optimize(
            lambda trial: self._objective(trial, trial_params, x, y),
            n_trials=25,
        )

        self.model = self.model.__class__(
            **study.best_trial.params,
            random_state=42,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        with parallel_backend("threading"):
            self.model.fit(x, y)
