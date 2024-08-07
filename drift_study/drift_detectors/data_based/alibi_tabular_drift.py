from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from alibi_detect.cd import TabularDrift as InternalTabularDrift
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NotFittedDetectorException,
)


class TabularAlibiDrift(DriftDetector):
    def __init__(
        self,
        x_metadata: pd.DataFrame,
        p_val: float = 0.05,
        correction: str = "bonferroni",
        alternative: str = "two-sided",
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            x_metadata=x_metadata,
            p_val=p_val,
            correction=correction,
            alternative=alternative,
            **kwargs,
        )

        self.x_metadata = x_metadata
        self.p_val = p_val
        self.correction = correction
        self.alternative = alternative

        self.x_test = pd.DataFrame()
        self.drift_detector = None
        self.window_size = None

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        categories_per_feature: Dict[int, Optional[List[int]]] = {}

        for index, row in self.x_metadata.iterrows():
            if row["type"] == "cat":
                categories_per_feature[int(index)] = None
        self.drift_detector = InternalTabularDrift(
            np.array(x),
            p_val=self.p_val,
            correction=self.correction,
            alternative=self.alternative,
            categories_per_feature=categories_per_feature,
        )
        self.x_test = x
        self.window_size = len(x)

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

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "p_val": trial.suggest_float("p_val", 1e-4, 0.1),
            "correction": trial.suggest_categorical(
                "correction", ["bonferroni", "fdr"]
            ),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "p_val": 0.05,
            "correction": "bonferroni",
        }

    def needs_model(self) -> bool:
        return False


detectors: Dict[str, Type[DriftDetector]] = {
    "tabular_alibi": TabularAlibiDrift,
}
