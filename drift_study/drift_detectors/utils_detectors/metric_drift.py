import numpy as np
from mlc.metrics.metric_factory import create_metric
from mlc.metrics.metrics import PredClassificationMetric


class MetricDrift:
    def __init__(
        self,
        drift_detector,
        metric_conf,
        **kwargs,
    ) -> None:
        self.drift_detector = drift_detector
        self.metric_conf = metric_conf
        self.metric = create_metric(metric_conf)

    def fit(self, y, y_scores, **kwargs):
        if y_scores is None:
            self.drift_detector.fit(y=y, y_scores=y_scores, **kwargs)
        else:
            metric = self.compute_metric(y, y_scores)
            self.drift_detector.fit(
                metric=metric, y=y, y_scores=y_scores, **kwargs
            )

    def update(self, y, y_scores, **kwargs):
        metric = self.compute_metric(y, y_scores)
        return self.drift_detector.update(
            metric=metric, y=y, y_scores=y_scores, **kwargs
        )

    def compute_metric(self, y, y_scores):

        if np.isscalar(y):
            return self.compute_metric(np.array([y]), np.array([y_scores]))[0]
        y_scores_l = y_scores
        if isinstance(self.metric, PredClassificationMetric):
            if len(y_scores_l.shape) > 1:
                y_scores_l = np.argmax(y_scores_l, axis=1)
            else:
                y_scores_l = np.argmax(y_scores_l)
        return self.metric.compute(y, y_scores_l)

    def needs_label(self):
        return True
