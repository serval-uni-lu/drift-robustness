from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model
from mlc.models.pipeline import Pipeline
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from torch import nn

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NotFittedDetectorException,
)
from drift_study.model_arch.lazy_pipeline import LazyPipeline
from drift_study.model_arch.sklearn_opt import TimeOptimizer


class AriesDrift(DriftDetector):
    def __init__(
        self,
        drift_detector: DriftDetector,
        accuracy_type: int = 0,
        section_num: int = 50,
        **kwargs: Dict[str, Any],
    ) -> None:

        super().__init__(
            drift_detector=drift_detector,
            acc=accuracy_type,
            section_num=section_num,
            **kwargs,
        )
        self.drift_detector = drift_detector
        self.pred_hist: Optional[PredHist] = None
        self.model: Optional[Model] = None
        self.base_acc = -1
        self.accuracy_type = accuracy_type
        self.section_num = section_num

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        self.model = model
        if isinstance(self.model, LazyPipeline):
            self.model._pipeline_load()
            self.model = self.model.pipeline

        if not isinstance(self.model, Pipeline):
            raise NotImplementedError
        if isinstance(self.model[-1].model, RandomForestClassifier):
            self.section_num = self.model[-1].model.n_estimators
        if isinstance(self.model[-1].model, TimeOptimizer):
            self.section_num = self.model[-1].model.model.n_estimators
        self.pred_hist = build_hist(
            self.model, x, y, section_num=self.section_num
        )
        self.base_acc = (self.model.predict(x) == y).mean()
        self.drift_detector.fit(x=x, t=t, y=y, y_scores=y_scores, model=model)

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        x = pd.DataFrame(x)

        if self.pred_hist is None:
            raise NotFittedDetectorException

        accs = deep_et_estimation(
            x,
            model=self.model,
            ori_hist=self.pred_hist,
            base_acc=self.base_acc,
            section_num=self.section_num,
        )
        estimated_acc = accs[self.accuracy_type]

        simulate = np.concatenate(
            [
                np.zeros(int(np.ceil((1 - estimated_acc) * len(x)))),
                np.ones(int(np.ceil(estimated_acc * len(x)))),
            ]
        )[: len(x)]
        new_y = np.ones(len(simulate))
        new_x = pd.DataFrame.from_dict({"simulated_error": simulate})
        new_y_scores = simulate
        np.random.shuffle(simulate)
        is_drift, is_warning, metrics = self.drift_detector.update(
            new_x, t, new_y, new_y_scores
        )
        for i, e in enumerate(accs):
            metrics[f"aries_acc_{i}"] = [e]

        return is_drift, is_warning, metrics

    def needs_model(self) -> bool:
        return True

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if trial_params["model_type"] == "random_forest":
            return {}
        return {
            "section_num": trial.suggest_int("section_num", 10, 100),
        }

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        if trial_params["model_type"] == "random_forest":
            return {}
        return {
            "section_num": 50,
        }


@dataclass
class PredHist:
    mode: npt.NDArray[np.int_]
    size: npt.NDArray[np.int_]
    proba: Union[None, ArrayLike]


def deep_et_estimation(
    x: pd.DataFrame,
    model: Model,
    ori_hist: PredHist,
    base_acc: float,
    section_num: int = 50,
) -> Tuple[float, float, float]:

    new_hist = build_hist(model, x, section_num=section_num)

    _, ori_idx, new_idx = np.intersect1d(
        ori_hist.mode, new_hist.mode, return_indices=True
    )

    combined_hist = PredHist(
        new_hist.mode[new_idx], new_hist.size[new_idx], ori_hist.proba[ori_idx]
    )
    acc1 = (
        np.sum(combined_hist.proba * combined_hist.size)
        / combined_hist.size.sum()
    )
    factor = (new_hist.size[-1] / new_hist.size.sum()) / (
        ori_hist.size[-1] / ori_hist.size.sum()
    )
    acc2 = min(1.0, (factor * base_acc))

    estimated_acc = (acc1 + acc2) / 2
    if estimated_acc > 1.0:
        print(estimated_acc)
    return estimated_acc, acc1, acc2


def predict_n_times(
    model: Model, x: pd.DataFrame, n_times: int
) -> npt.NDArray[np.int_]:
    model[-1].train()
    x_l = pd.concat([x] * n_times, ignore_index=True)
    prediction = model.predict(x_l)
    prediction = prediction.reshape(
        (n_times, int(prediction.shape[0] / n_times), *prediction.shape[1:])
    )
    model[-1].eval()
    return prediction


def compute_mode(model, x, section_num) -> npt.NDArray[np.int_]:
    if isinstance(model[-1].model, nn.Module):
        # Make many prediction
        t_predictions_list = predict_n_times(model, x, section_num)
        # Calculate the mode
        mode_list = []
        for _ in range(len(x)):
            mode_num = stats.mode(
                t_predictions_list[:, _ : (_ + 1)].reshape(
                    -1,
                )
            )[1][0]
            mode_list.append(mode_num)
        return np.asarray(mode_list)
    elif isinstance(model[-1].model, RandomForestClassifier):
        prediction = model.predict_proba(x)
        prediction = np.max(prediction, axis=1)
        mode_list = np.round(prediction * model[-1].model.n_estimators).astype(
            int
        )
        return mode_list

    elif isinstance(model[-1].model, TimeOptimizer):
        prediction = model.predict_proba(x)
        prediction = np.max(prediction, axis=1)
        mode_list = np.round(
            prediction * model[-1].model.model.n_estimators
        ).astype(int)
        return mode_list

    else:
        raise NotImplementedError


def build_hist(
    model: Model, x: pd.DataFrame, y=None, section_num=50
) -> PredHist:

    # Decide whether to predict the probability of a region
    # based on true accuracy.
    # Usually compute at train time but not at test time,
    # because of true label availability.
    build_proba = y is not None

    # Prepare objects
    if build_proba:
        y = y.reshape(1, -1)[0]

    # Original predictions
    if build_proba:
        y_pred = model.predict(x)
        y_correct_idx = np.where(y_pred == y)[0]
    else:
        y_correct_idx = None

    mode_list = compute_mode(model, x, section_num)

    # Compute mode histogram
    hist_mode = []
    hist_size = []
    hist_correct = []
    for i in range(1, section_num + 1):
        consistent_idxs = np.where(mode_list == i)[0]
        if len(consistent_idxs) > 0:
            hist_mode.append(i)
            hist_size.append(len(consistent_idxs))
            if build_proba:
                current_correct_idx = np.intersect1d(
                    consistent_idxs, y_correct_idx
                )
                hist_correct.append(len(current_correct_idx))

    np_hist_mode = np.array(hist_mode)
    np_hist_size = np.array(hist_size)
    if build_proba:
        np_hist_correct = np.array(hist_correct)
        np_hist_proba_correct = np_hist_correct / np_hist_size
    else:
        np_hist_proba_correct = None
    pred_hist = PredHist(np_hist_mode, np_hist_size, np_hist_proba_correct)

    return pred_hist


detectors: Dict[str, Type[DriftDetector]] = {"aries": AriesDrift}
