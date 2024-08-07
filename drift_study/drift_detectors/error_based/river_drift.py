from abc import ABC
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.metrics.metric_factory import create_metric
from mlc.models.model import Model
from river import drift
from river.base import DriftDetector as RiverDriftDetector

from drift_study.drift_detectors.drift_detector import DriftDetector
from drift_study.utils.metric import compute_metric


class RiverDrift(DriftDetector, ABC):
    def __init__(
        self,
        internal_detector_cls: Type[RiverDriftDetector],
        metric_conf: Dict[str, Any],
        internal_args: Union[dict, None] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            internal_detector_cls=internal_detector_cls,
            internal_args=internal_args,
            metric_conf=metric_conf,
            **kwargs,
        )
        self.internal_detector_cls = internal_detector_cls
        self.metric_conf = metric_conf

        self.metric = create_metric(metric_conf)

        self.internal_args = internal_args
        if self.internal_args is None:
            self.internal_args = {}
        self.drift_detector = None

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        self.drift_detector = self.internal_detector_cls(**self.internal_args)

    def _update_one(self, metric: Union[float, int]):
        self.drift_detector.update(metric)
        in_drift = self.drift_detector.drift_detected
        return in_drift, False, pd.DataFrame()

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ):
        metric = compute_metric(self.metric, y, y_scores)
        if not hasattr(metric, "__len__"):
            return self._update_one(metric)
        else:
            was_drift, was_warning = False, False
            for i in np.arange(len(metric)):
                metric_0 = metric[i]
                in_drift, in_warning, _ = self._update_one(metric_0)
                was_drift = was_drift or in_drift
                was_warning = was_warning or in_warning
            return was_drift, was_warning, pd.DataFrame()

    def needs_model(self) -> bool:
        return True


class AdwinDrift(RiverDrift):
    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        metric_conf=None,
        **kwargs: Dict[str, Any],
    ) -> None:
        internal_args = {
            "delta": delta,
            "clock": clock,
            "max_buckets": max_buckets,
            "min_window_length": min_window_length,
            "grace_period": grace_period,
        }
        if metric_conf is None:
            metric_conf = {"name": "class_error"}
        super().__init__(drift.ADWIN, metric_conf, internal_args, **kwargs)

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "delta": trial.suggest_float("delta", 1e-6, 1 - 1e-6),
            "grace_period": trial.suggest_int("grace_period", 10, 1000),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "delta": 0.002,
            "grace_period": 10,
        }


class DdmDrift(RiverDrift):
    def __init__(
        self,
        warm_start: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
        **kwargs: Dict[str, Any],
    ):
        internal_args = {
            "warm_start": warm_start,
            "warning_threshold": warning_threshold,
            "drift_threshold": drift_threshold,
        }
        super().__init__(
            drift.DDM, {"name": "class_error"}, internal_args, **kwargs
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "warm_start": trial.suggest_int("warm_start", 30, 1000),
            "drift_threshold": trial.suggest_float(
                "drift_threshold", 3e-1, 3e1
            ),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "warm_start": 30,
            "drift_threshold": 3.0,
        }


class EddmDrift(RiverDrift):
    def __init__(
        self,
        warm_start: int = 30,
        alpha: float = 0.95,
        beta: float = 0.9,
        **kwargs: Dict[str, Any],
    ):
        internal_args = {
            "warm_start": warm_start,
            "alpha": alpha,
            "beta": beta,
        }
        super().__init__(
            drift.EDDM, {"name": "class_error"}, internal_args, **kwargs
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        beta = trial.suggest_float("beta", 0, 1)
        alpha = beta
        return {
            "warm_start": trial.suggest_int("warm_start", 30, 1000),
            "beta": beta,
            "alpha": alpha,
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "warm_start": 30,
            "alpha": 0.95,
            "beta": 0.9,
        }


class HdddmADrift(RiverDrift):
    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        two_sided_test: bool = False,
        **kwargs: Dict[str, Any],
    ):
        internal_args = {
            "drift_confidence": drift_confidence,
            "warning_confidence": warning_confidence,
            "two_sided_test": two_sided_test,
        }
        super().__init__(
            drift.HDDM_A, {"name": "class_error"}, internal_args, **kwargs
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "drift_confidence": trial.suggest_float("drift_confidence", 0, 1),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "drift_confidence": 0.001,
        }


class HdddmWDrift(RiverDrift):
    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        lambda_val: float = 0.05,
        two_sided_test: bool = False,
        **kwargs: Dict[str, Any],
    ):
        internal_args = {
            "drift_confidence": drift_confidence,
            "warning_confidence": warning_confidence,
            "lambda_val": lambda_val,
            "two_sided_test": two_sided_test,
        }
        super().__init__(
            drift.HDDM_W, {"name": "class_error"}, internal_args, **kwargs
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "drift_confidence": trial.suggest_float("drift_confidence", 0, 1),
            "lambda_val": trial.suggest_float("lambda_val", 0, 1),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {"drift_confidence": 0.001, "lambda_val": 0.05}


class KswinDrift(RiverDrift):
    def __init__(
        self,
        alpha: float = 0.005,
        ks_window_size: int = 100,
        stat_size: int = 30,
        metric_conf=None,
        seed: int = 42,
        **kwargs,
    ):
        internal_args = {
            "alpha": alpha,
            "window_size": ks_window_size,
            "stat_size": stat_size,
            "seed": seed,
        }
        if metric_conf is None:
            metric_conf = {"name": "class_error"}
        super().__init__(drift.KSWIN, metric_conf, internal_args, **kwargs)

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        ks_window_size = trial.suggest_int("ks_window_size", 50, int(1e3))
        stat_size = int(ks_window_size / 3)
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 2e-2),
            "stat_size": stat_size,
            "ks_window_size": ks_window_size,
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "alpha": 0.005,
            "stat_size": 30,
            "ks_window_size": 100,
        }


class PageHinkleyDrift(RiverDrift):
    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
        mode: str = "both",
        metric_conf=None,
        **kwargs,
    ):
        internal_args = {
            "min_instances": min_instances,
            "delta": delta,
            "threshold": threshold,
            "alpha": alpha,
            "mode": mode,
        }
        if metric_conf is None:
            metric_conf = {"name": "class_error"}
        super().__init__(
            drift.PageHinkley, metric_conf, internal_args, **kwargs
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "min_instances": trial.suggest_int("min_instances", 30, int(1e3)),
            "delta": trial.suggest_float("delta", 1e-6, 1e-2),
            "threshold": trial.suggest_float("threshold", 1, 5e2),
            "alpha": trial.suggest_float("alpha", 1 - 1e-1, 1 - 1e-9),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "min_instances": 30,
            "delta": 0.005,
            "threshold": 50.0,
            "alpha": 1 - 0.0001,
        }


detectors: Dict[str, Type[DriftDetector]] = {
    "adwin": AdwinDrift,
    "ddm": DdmDrift,
    "eddm": EddmDrift,
    "hddm_a": HdddmADrift,
    "hddm_w": HdddmWDrift,
    "kswin": KswinDrift,
    "page_hinkley": PageHinkleyDrift,
}
