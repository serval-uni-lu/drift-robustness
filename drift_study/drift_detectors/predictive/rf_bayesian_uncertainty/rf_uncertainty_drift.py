import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas
import pandas as pd
from mlc.models.model import Model
from mlc.models.pipeline import Pipeline

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NoModelException,
    NotFittedDetectorException,
)
from drift_study.model_arch.lazy_pipeline import LazyPipeline

from .rf_uncertainty import RandomForestClassifierWithUncertainty

warnings.simplefilter(action="ignore", category=FutureWarning)

UNCERTAINTY_TYPE = {"total": 0, "epistemic": 1, "aleatoric": 2}


class RfUncertaintyDrift(DriftDetector):
    def __init__(
        self,
        drift_detector: DriftDetector,
        uncertainty_type: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            drift_detector=drift_detector,
            uncertainty_type=uncertainty_type,
            **kwargs,
        )
        self.drift_detector = drift_detector
        self.uncertainty_type = uncertainty_type
        self.rf_uncertainty: Optional[
            RandomForestClassifierWithUncertainty
        ] = None
        self.rf = None
        self.model: Optional[Model] = None

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        if model is None:
            raise NoModelException

        if isinstance(model, LazyPipeline):
            model._pipeline_load()
            model = model.pipeline

        if not isinstance(model, Pipeline):
            raise NotImplementedError("Model is expected to be a Pipeline")

        internal_model = model[-1]
        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
        self.rf = internal_model
        self.model = model

        self.rf_uncertainty = RandomForestClassifierWithUncertainty(
            random_forest=self.rf
        )
        self.rf_uncertainty.fit(self.model.transform(x), y)
        (
            _,
            uncertainties,
        ) = self.rf_uncertainty.predict_proba_with_uncertainty(
            self.model.transform(x)
        )
        uncertainties = uncertainties[UNCERTAINTY_TYPE[self.uncertainty_type]]
        x_new = pd.DataFrame.from_dict({"uncertainty": uncertainties})
        y_scores_new = np.array(uncertainties)
        self.drift_detector.fit(
            x=x_new,
            t=t,
            y=y,
            y_scores=y_scores_new,
            model=model,
        )

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        if (self.model is None) or (self.rf_uncertainty is None):
            raise NotFittedDetectorException
        if not isinstance(self.model, Pipeline):
            raise NotImplementedError("Model is expected to be a Pipeline")
        x = pandas.DataFrame(x)
        _, uncertainties = self.rf_uncertainty.predict_proba_with_uncertainty(
            self.model.transform(x)
        )

        uncertainties = uncertainties[UNCERTAINTY_TYPE[self.uncertainty_type]]
        x_new = pd.DataFrame.from_dict({"uncertainty": uncertainties})
        y_scores_new = np.array(uncertainties)

        is_drift, is_warning, metrics = self.drift_detector.update(
            x=x_new, t=t, y=y, y_scores=y_scores_new
        )
        metrics[f"{self.uncertainty_type}_uncertainty"] = uncertainties
        return is_drift, is_warning, metrics

    def needs_model(self) -> bool:
        return True

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "uncertainty_type": trial.suggest_categorical(
                "uncertainty_type", ["total", "epistemic", "aleatoric"]
            )
        }

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        return {"uncertainty_type": "total"}


detectors: Dict[str, Type[DriftDetector]] = {
    "rf_uncertainty": RfUncertaintyDrift
}
