import abc
import logging
from abc import ABC
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model


class NotFittedDetectorException(Exception):
    """Raised when the DriftDetector is not fitted but update is called."""

    pass


class NoModelException(Exception):
    """Raised when the DriftDetector needs the model but does not find one."""

    pass


class DriftDetector(ABC):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abc.abstractmethod
    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        pass

    @abc.abstractmethod
    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        pass

    @staticmethod
    @abc.abstractmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def needs_model(self) -> bool:
        pass

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        logger = logging.getLogger(__name__)
        logger.warning("Default parameters not set.")
        return None
