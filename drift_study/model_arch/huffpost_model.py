from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from mlc.models.torch_models import BaseModelTorch

from drift_study.model_arch.article_networks import ArticleNetwork


class HuffpostModel(BaseModelTorch):
    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 1,
        early_stopping_rounds: int = 0,
        learning_rate: float = 2e-5,
        **kwargs: Any,
    ) -> None:
        name = "huffpost_bert"
        objective = "classification"
        super().__init__(
            name,
            objective,
            batch_size,
            epochs,
            early_stopping_rounds,
            learning_rate,
            val_batch_size=512,
            class_weight=None,
            is_text=True,
            weight_decay=1e-2,
            **kwargs,
        )

        self.model = ArticleNetwork(num_classes=41)
        for p in self.model.model[0].parameters():
            p.requires_grad = True
        self.to_device()

    def fit(
        self,
        x: npt.NDArray[np.float_],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        x_val: Optional[npt.NDArray[np.float_]] = None,
        y_val: Optional[
            Union[npt.NDArray[np.int_], npt.NDArray[np.float_]]
        ] = None,
        reset_weight: bool = False,
    ):
        x = x.reshape((x.shape[0], -1, 2))
        if x_val is not None:
            x_val = x.reshape((x_val.shape[0], -1, 2))
        super().fit(x, y, x_val, y_val, reset_weight)

    def predict_helper(
        self, x: npt.NDArray[np.float_], load_all_gpu: bool = True
    ) -> npt.NDArray[np.float_]:
        x = x.reshape((x.shape[0], -1, 2))
        return super().predict_helper(x, load_all_gpu)


models = [
    ("huffpost_bert", HuffpostModel),
]
