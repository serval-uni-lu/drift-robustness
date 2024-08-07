from typing import Dict, Type

from drift_study.drift_detectors.drift_detector import DriftDetector

from . import rf_uncertainty_drift

detectors: Dict[str, Type[DriftDetector]] = {**rf_uncertainty_drift.detectors}
