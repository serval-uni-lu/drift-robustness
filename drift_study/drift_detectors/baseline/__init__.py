from typing import Dict, Type

from drift_study.drift_detectors.baseline import (
    manual_index,
    no_detection,
    periodic_drift,
)
from drift_study.drift_detectors.drift_detector import DriftDetector

detectors: Dict[str, Type[DriftDetector]] = {
    **no_detection.detectors,
    **periodic_drift.detectors,
    **manual_index.detectors,
}
