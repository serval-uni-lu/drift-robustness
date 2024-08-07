from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import DriftDetector


class PeriodicDrift(DriftDetector):
    def __init__(self, period: int, **kwargs: Dict[str, Any]) -> None:
        super().__init__(period=period, **kwargs)
        self.period = period
        self.counter = 0

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        self.counter = 0

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        if len(x.shape) == 1:
            self.counter += 1
        elif len(x.shape) == 2:
            self.counter += x.shape[0]

        if self.counter < self.period:
            return False, False, pd.DataFrame()
        else:

            return True, True, pd.DataFrame()

    def needs_model(self) -> bool:
        return False

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "period": trial.suggest_int(
                "period",
                trial_params["period"]["min"],
                trial_params["period"]["max"],
            )
        }
        return params

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        params = {"period": trial_params["period"]["min"]}
        return params


detectors = {"periodic": PeriodicDrift}
