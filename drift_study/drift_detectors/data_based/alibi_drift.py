from abc import ABC
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from alibi_detect.base import DriftConfigMixin
from alibi_detect.cd import LearnedKernelDrift, LSDDDrift, MMDDrift
from alibi_detect.cd import SpotTheDiffDrift as AlibiSpotTheDiffDrift
from alibi_detect.utils.pytorch import GaussianRBF
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NotFittedDetectorException,
)


class AlibiDrift(DriftDetector, ABC):
    def __init__(
        self,
        window_size: int,
        internal_detector_cls: Type[DriftConfigMixin],
        internal_args: Union[Dict[str, Any], None] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            window_size=window_size,
            internal_detector_cls=internal_detector_cls,
            internal_args=internal_args,
            **kwargs,
        )
        self.internal_detector_cls = internal_detector_cls

        self.internal_args = internal_args
        if self.internal_args is None:
            self.internal_args = {}
        self.window_size = window_size
        self.x_test = pd.DataFrame()

        self.drift_detector: Optional[Type[DriftConfigMixin]] = None

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        self.drift_detector = self.internal_detector_cls(
            x_ref=x.to_numpy().astype(np.double),
            backend="pytorch",
            **self.internal_args,
        )
        self.x_test = x

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        x = pd.DataFrame(x)
        self.x_test = pd.concat([self.x_test, x])
        self.x_test = self.x_test.iloc[-self.window_size :]

        if self.drift_detector is None:
            raise NotFittedDetectorException
        else:
            out = self.drift_detector.predict(
                self.x_test.to_numpy().astype(np.double)
            )
            is_drift = out["data"]["is_drift"] > 0
            is_warning = is_drift
            data = {}
            for e in out["data"].keys():
                data[e] = [out["data"][e]]
            return is_drift, is_warning, pd.DataFrame.from_dict(data)

    def needs_model(self) -> bool:
        return False


class MmdDrift(AlibiDrift):
    def __init__(
        self,
        window_size: int,
        p_val: float = 0.05,
        n_permutations: int = 100,
        **kwargs: Dict[str, Any],
    ):
        internal_args = {"p_val": p_val, "n_permutations": n_permutations}
        super().__init__(
            window_size, MMDDrift, internal_args=internal_args, **kwargs
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "p_val": trial.suggest_float("p_val", 1e-4, 0.1),
            "n_permutations": trial.suggest_int("n_permutations", 10, 1000),
        }


class LsddDrift(AlibiDrift):
    def __init__(
        self,
        window_size: int,
        p_val: float = 0.05,
        n_permutations: int = 100,
        **kwargs: Dict[str, Any],
    ):
        internal_args = {"p_val": p_val, "n_permutations": n_permutations}
        super().__init__(
            window_size, LSDDDrift, internal_args=internal_args, **kwargs
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "p_val": trial.suggest_float("p_val", 1e-4, 0.5),
            "n_permutations": trial.suggest_int("n_permutations", 10, 1000),
        }


class MmdLkDrift(AlibiDrift):
    def __init__(
        self,
        window_size: int,
        p_val: float = 0.05,
        n_permutations: int = 100,
        **kwargs: Dict[str, Any],
    ) -> None:
        internal_args = {
            "p_val": p_val,
            "n_permutations": n_permutations,
            "kernel": GaussianRBF(),
        }
        super().__init__(
            window_size,
            LearnedKernelDrift,
            internal_args=internal_args,
            **kwargs,
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "p_val": trial.suggest_float("p_val", 1e-4, 0.1),
            "n_permutations": trial.suggest_int("n_permutations", 10, 1000),
        }


class SpotTheDiffDrift(AlibiDrift):
    def __init__(
        self,
        window_size: int,
        p_val: float = 0.05,
        n_diffs: int = 100,
        **kwargs: Dict[str, Any],
    ):
        internal_args = {"p_val": p_val, "n_diffs": n_diffs}
        super().__init__(
            window_size,
            AlibiSpotTheDiffDrift,
            internal_args=internal_args,
            **kwargs,
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "p_val": trial.suggest_float("p_val", 1e-4, 0.1),
            "n_diffs": trial.suggest_int("n_diffs", 1, 10),
        }


detectors: Dict[str, Type[DriftDetector]] = {
    "mmd": MmdDrift,
    "mmd_lk": MmdLkDrift,
    "spot_the_diff": SpotTheDiffDrift,
    "lsdd": LsddDrift,
}
