from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import DriftDetector


class ManualIndex(DriftDetector):
    def __init__(
        self,
        end_idx: int,
        detect_idxs: List[int],
        **kwargs: Dict[str, Any],
    ) -> None:
        # No drift does not require parameters
        # Used to be compliant with others drift detectors
        self.end_idx = end_idx
        self.detect_idxs = detect_idxs
        self.count = end_idx
        super().__init__(
            start_index=end_idx, detect_idxs=detect_idxs, **kwargs
        )

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
        if self.count in self.detect_idxs:
            return True, False, pd.DataFrame()
        # End idx is the first test, so we increment after checking
        self.count += 1
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
    "manual_index": ManualIndex,
}
