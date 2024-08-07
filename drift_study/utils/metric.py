from typing import Union

import numpy as np
import numpy.typing as npt
from mlc.metrics.metric import Metric
from mlc.metrics.metrics import PredClassificationMetric


def compute_metric(
    metric: Metric,
    y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
    y_scores: Union[npt.NDArray[np.float_]],
) -> Union[npt.NDArray[np.int_], npt.NDArray[np.float_], float, int]:
    if np.isscalar(y):
        return compute_metric(metric, np.array([y]), np.array([y_scores]))[0]
    y_scores_l = y_scores
    if isinstance(metric, PredClassificationMetric):
        if len(y_scores_l.shape) > 1:
            y_scores_l = np.argmax(y_scores_l, axis=1)
        else:
            y_scores_l = np.argmax(y_scores_l)
    return metric.compute(y, y_scores_l)
