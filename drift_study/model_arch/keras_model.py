from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from mlc.models.tf_models import TfModel


class NnElectricity(TfModel):
    def __init__(self, name: str = "nn_electricity", **kwargs: Dict[str, Any]):
        super().__init__(name=name, objective="binary", **kwargs)
        self.model = None

    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.int_],
        x_val: Union[None, npt.NDArray[np.float64]] = None,
        y_val: Union[None, npt.NDArray[np.int_]] = None,
    ) -> None:
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        self.model.fit(
            x, y, class_weight={0: y.mean(), 1: 1 - y.mean()}, epochs=10
        )


models = [("nn_electricity", NnElectricity)]
