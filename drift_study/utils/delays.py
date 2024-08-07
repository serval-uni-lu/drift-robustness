from dataclasses import dataclass

import pandas as pd

from drift_study.drift_detectors import DriftDetector


@dataclass
class Delays:
    label: pd.Timedelta
    drift_detector: pd.Timedelta
    ml_model: pd.Timedelta


def get_delays(run_config: dict, drift_detector: DriftDetector) -> Delays:
    delays = run_config.get("delays")
    label_delay = pd.Timedelta(delays.get("label"))
    retraining_delay = pd.Timedelta(delays.get("retraining"))

    if int(delays.get("drift")) != 0:
        raise NotImplementedError

    model_delay = pd.Timedelta(label_delay + retraining_delay)
    if drift_detector.needs_model():
        drift_detection_delay = model_delay
    else:
        drift_detection_delay = pd.Timedelta(0)

    delays = Delays(label_delay, drift_detection_delay, model_delay)
    return delays
