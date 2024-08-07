from typing import Dict, Type

from drift_study.drift_detectors.data_based import (
    alibi_drift,
    alibi_tabular_drift,
    evidently_drift,
    pca_cd,
)
from drift_study.drift_detectors.drift_detector import DriftDetector

detectors: Dict[str, Type[DriftDetector]] = {
    **alibi_drift.detectors,
    **alibi_tabular_drift.detectors,
    **evidently_drift.detectors,
    **pca_cd.detectors,
}
