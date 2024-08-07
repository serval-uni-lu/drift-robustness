from typing import Dict, Type

from drift_study.drift_detectors.drift_detector import DriftDetector
from drift_study.drift_detectors.predictive import (
    aries_drift,
    rf_bayesian_uncertainty,
)

detectors: Dict[str, Type[DriftDetector]] = {
    **rf_bayesian_uncertainty.detectors,
    **aries_drift.detectors,
}
