from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import DriftDetector


class NBatchDrift(DriftDetector):
    def __init__(
        self,
        drift_detector: DriftDetector,
        batch_size: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            drift_detector=drift_detector, batch_size=batch_size, **kwargs
        )
        self.drift_detector = drift_detector
        self.batch_size = batch_size
        self.counter = 0
        self.mem: List[Dict[str, Any]] = []

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        self.drift_detector.fit(x, t, y, y_scores, model)
        self.mem = []
        self.counter = 0

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        self.mem.append({"x": x, "t": t, "y": y, "y_scores": y_scores})
        self.counter += 1
        if self.counter >= self.batch_size:
            mem = {k: [dic[k] for dic in self.mem] for k in self.mem[0]}
            mem["x"] = pd.concat(mem["x"])
            if isinstance(mem["t"], pd.Series):
                mem["t"] = pd.concat(mem["t"])
            else:
                mem["t"] = np.concatenate(mem["t"], axis=0)
            mem["y"] = np.concatenate(mem["y"], axis=0)
            mem["y_scores"] = np.concatenate(mem["y_scores"], axis=0)

            self.mem = []
            self.counter = 0
            return self.drift_detector.update(**mem)

        return False, False, pd.DataFrame()

    def needs_model(self) -> bool:
        return self.drift_detector.needs_model()

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


detectors: Dict[str, Type[DriftDetector]] = {"n_batch": NBatchDrift}
