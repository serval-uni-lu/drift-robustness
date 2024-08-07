import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NotFittedDetectorException,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


class EvidentlyDrift(DriftDetector):
    def __init__(
        self,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        num_threshold: float = 0.05,
        cat_threshold: float = 0.05,
        drift_share: float = 0.5,
        fit_first_update: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            num_threshold=num_threshold,
            cat_threshold=cat_threshold,
            drift_share=drift_share,
            **kwargs,
        )
        self.window_size = None
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.drift_share = drift_share
        self.num_threshold = num_threshold
        self.cat_threshold = cat_threshold

        self.drift_detector: Optional[Report] = None
        self.column_mapping = ColumnMapping()

        self.x_ref = pd.DataFrame()
        self.x_last = pd.DataFrame()
        self.fit_first_update = fit_first_update
        self.is_first_update = True

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:

        metric = DataDriftPreset(
            drift_share=self.drift_share,
            num_stattest="wasserstein",
            num_stattest_threshold=self.num_threshold,
            cat_stattest="jensenshannon",
            cat_stattest_threshold=self.cat_threshold,
        )
        self.drift_detector = Report(metrics=[metric])
        self.column_mapping = ColumnMapping()
        self.column_mapping.numerical_features = self.numerical_features
        self.column_mapping.categorical_features = self.categorical_features
        self.x_ref = x
        self.x_last = x
        self.is_first_update = True
        self.window_size = len(x)

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:

        if self.drift_detector is None:
            raise NotFittedDetectorException
        else:
            if self.fit_first_update and self.is_first_update:
                self.window_size = len(x)
                self.fit(x, t, y, y_scores, None)

            self.x_last = pd.concat(
                [
                    self.x_last,
                    pd.DataFrame(x, columns=self.x_last.columns),
                ]
            )[-self.window_size :]

            self.drift_detector.run(
                reference_data=self.x_ref,
                current_data=self.x_last,
                column_mapping=self.column_mapping,
            )
            report = self.drift_detector.as_dict()

            in_drift = report["metrics"][0]["result"]["dataset_drift"]
            in_warning = (
                report["metrics"][0]["result"]["number_of_drifted_columns"] > 0
            )

            self.is_first_update = False
            return (
                in_drift,
                in_warning,
                pd.DataFrame(),
            )

    def needs_model(self) -> bool:
        return False

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "cat_threshold": trial.suggest_float(
                "categorical_threshold", 1e-6, 0.5
            ),
            "num_threshold": trial.suggest_float(
                "numerical_threshold", 1e-6, 0.5
            ),
            "drift_share": trial.suggest_float("drift_share", 1e-6, 1),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "num_threshold": 0.05,
            "cat_threshold": 0.05,
            "drift_share": 0.5,
        }


detectors: Dict[str, Type[DriftDetector]] = {"evidently": EvidentlyDrift}
