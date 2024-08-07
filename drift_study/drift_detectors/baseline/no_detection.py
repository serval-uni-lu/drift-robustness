from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import DriftDetector


class NoDetection(DriftDetector):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        # No drift does not require parameters
        # Used to be compliant with others drift detectors
        super().__init__(**kwargs)

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        # No drift does not require fitting
        # Used to be compliant with others drift detectors
        pass

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        return False, False, pd.DataFrame()

    def needs_model(self) -> bool:
        return False

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {}


detectors: Dict[str, Type[DriftDetector]] = {
    "no_detection": NoDetection,
}
