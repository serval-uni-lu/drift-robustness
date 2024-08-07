from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from mlc.models.torch_models import BaseModelTorch
from wildtime.networks.yearbook import YearbookNetwork


class YearBookModel(BaseModelTorch):
    def __init__(
        self,
        batch_size: int = 512,
        epochs: int = 10,
        early_stopping_rounds: int = 2,
        learning_rate: float = 0.001,
        **kwargs: Any,
    ) -> None:
        name = "yearbook_conv_nn"
        objective = "binary"
        super().__init__(
            name,
            objective,
            batch_size,
            epochs,
            early_stopping_rounds,
            learning_rate,
            class_weight="balanced",
            force_device="cpu",
            **kwargs,
        )

        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)

        args = Struct(**{"method": None})
        self.model = YearbookNetwork(args, num_input_channels=1, num_classes=1)
        self.to_device()

    def fit(
        self,
        x: npt.NDArray[np.float_],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        x_val: Optional[npt.NDArray[np.float_]] = None,
        y_val: Optional[
            Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]
        ] = None,
        reset_weight: bool = True,
    ):
        x = x.reshape(-1, 1, 32, 32)
        if x_val is not None:
            x = x.reshape(-1, 32, 32)

        return super().fit(x, y, x_val, y_val, reset_weight)

    def predict_helper(
        self, x: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        x = x.reshape(-1, 1, 32, 32)
        return super().predict_helper(x)


models = [
    ("yearbook_conv_nn", YearBookModel),
]
