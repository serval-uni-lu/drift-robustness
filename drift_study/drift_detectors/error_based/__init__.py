from typing import Dict, Type

from drift_study.drift_detectors.drift_detector import DriftDetector
from drift_study.drift_detectors.error_based import river_drift

detectors: Dict[str, Type[DriftDetector]] = {
    **river_drift.detectors,
}
