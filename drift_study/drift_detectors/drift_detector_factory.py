from typing import Any, Dict, List, Union

from configutils.utils import merge_parameters

from drift_study.drift_detectors import get_drift_detectors


def get_drift_detector(
    drift_detectors_names: Union[str, List[str]], **kwargs: Dict[str, Any]
):

    if isinstance(drift_detectors_names, str):
        return get_drift_detectors(drift_detectors_names)(**kwargs)
    else:
        drift_detectors_names.reverse()
        detector = None
        for element in drift_detectors_names:
            local_detector_class = get_drift_detectors(element)
            detector = local_detector_class(**kwargs, drift_detector=detector)

        return detector


def get_drift_detector_from_conf(
    detectors, common_detectors_params: Dict[str, Any]
):

    detectors = detectors.copy()
    detectors.reverse()
    out_detector = None
    for detector in detectors:
        local_detector_class = get_drift_detectors(detector.get("name"))
        out_detector = local_detector_class(
            **merge_parameters(
                common_detectors_params, detector.get("params", {})
            ),
            drift_detector=out_detector,
        )
    return out_detector


def get_drift_detector_class_from_conf(detectors) -> List[Dict[str, Any]]:
    detectors = detectors.copy()
    out = []
    for detector in detectors:
        local_detector_class = get_drift_detectors(detector.get("name"))
        out.append(
            {
                "detector": local_detector_class,
                "params": detector.get("params", {}),
            }
        )

    return out
