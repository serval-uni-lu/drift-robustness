from typing import Any, Dict

from mlc.models.sk_models import SkModel
from sklearn import linear_model


class LogisticRegressionModel(SkModel):
    def __init__(
        self, name: str = "logistic_regression", **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(name=name, objective="binary", **kwargs)
        self.model = linear_model.LogisticRegression(
            n_jobs=-1, class_weight="balanced"
        )


class LinearRegressionModel(SkModel):
    def __init__(
        self, name: str = "linear_regression", **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(name=name, objective="regression", **kwargs)
        self.model = linear_model.LinearRegression(n_jobs=-1)


models = [
    ("logistic_regression", LogisticRegressionModel),
    ("linear_regression", LinearRegressionModel),
]
