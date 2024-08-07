from typing import Any, Dict, List, Type, Union

from drift_study.drift_detectors import (
    baseline,
    data_based,
    error_based,
    predictive,
    utils_detectors,
)
from drift_study.drift_detectors.drift_detector import DriftDetector

drift_detectors = {
    **baseline.detectors,
    **data_based.detectors,
    **error_based.detectors,
    **predictive.detectors,
    **utils_detectors.detectors,
}


def load_one(drift_detectors_name: str) -> Type[DriftDetector]:
    if drift_detectors_name in drift_detectors:
        return drift_detectors[drift_detectors_name]
    else:
        raise NotImplementedError(
            f"Drift detector '{drift_detectors_name}' is not available."
        )


def get_drift_detectors(
    drift_detectors_names: Union[str, List[str]]
) -> Union[Type[DriftDetector], List[Dict[str, Any]]]:

    if isinstance(drift_detectors_names, str):
        return load_one(drift_detectors_names)
    else:
        return [
            {
                "name": drift_detectors_name,
                "drift_detector": load_one(drift_detectors_name),
            }
            for drift_detectors_name in drift_detectors_names
        ]
