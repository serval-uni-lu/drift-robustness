from typing import Dict, Type

from drift_study.drift_detectors.drift_detector import DriftDetector
from drift_study.drift_detectors.utils_detectors import n_batch_drift

detectors: Dict[str, Type[DriftDetector]] = {
    **n_batch_drift.detectors,
}
