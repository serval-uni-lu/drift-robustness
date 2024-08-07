import logging
from multiprocessing import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset
from mlc.models.model import Model
from mlc.models.model_factory import get_model
from mlc.models.pipeline import Pipeline
from mlc.models.sk_models import SkModel
from mlc.transformers.tabular_transformer import TabTransformer
from numpy.typing import ArrayLike

from drift_study.drift_detectors import DriftDetector
from drift_study.drift_detectors.drift_detector_factory import (
    get_drift_detector_from_conf,
)
from drift_study.model_arch.lazy_pipeline import LazyPipeline
from drift_study.utils.date_sampler import sample_date
from drift_study.utils.delays import Delays
from drift_study.utils.drift_model import DriftModel
from drift_study.utils.io_utils import load_do_save_model


def update_dataset_name(
    dataset: Dataset, minority_share: Optional[float]
) -> None:
    if minority_share is not None:
        dataset.name = f"{dataset.name}_{str(minority_share)}"


def initialize(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Tuple[
    Dataset,
    Callable[[], Model],
    Callable[[int, int], DriftDetector],
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Loading dataset {config.get('dataset', {}).get('name')}")
    dataset = get_dataset(config.get("dataset"))
    x, y, t = dataset.get_x_y_t()
    x, y, t = sample_date(x, y, t, run_config.get("sampling_minority_share"))
    update_dataset_name(dataset, run_config.get("sampling_minority_share"))

    metadata = dataset.get_metadata(only_x=True)
    f_new_model = get_f_new_model(config, run_config, metadata)
    f_new_detector = get_f_new_detector(config, run_config, metadata)
    return dataset, f_new_model, f_new_detector, x, y, t


def get_f_new_detector(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    metadata_x: pd.DataFrame,
) -> Callable[[int, int], DriftDetector]:
    def f_new_detector(start_idx, end_idx) -> DriftDetector:
        drift_detector = get_drift_detector_from_conf(
            run_config.get("detectors"),
            get_common_detectors_params(
                config, metadata_x, start_idx, end_idx
            ),
        )
        return drift_detector

    return f_new_detector


def get_f_new_model(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    metadata_x: pd.DataFrame,
) -> Callable[[], Model]:
    def new_model() -> Model:
        model = get_model_l(config, run_config, metadata_x)
        model = LazyPipeline(
            Pipeline(
                steps=[
                    TabTransformer(
                        metadata=metadata_x, scale=True, one_hot_encode=True
                    ),
                    model,
                ]
            )
        )
        return model

    return new_model


def get_model_l(
    config: Dict[str, Any], run_config: Dict[str, Any], metadata: pd.DataFrame
) -> Model:

    model_class = get_model(run_config.get("model"))
    n_jobs = config.get("performance", {}).get("n_jobs", {}).get("model", None)
    model = model_class(
        # x_metadata=metadata,
        verbose=0,
        n_jobs=n_jobs,
        random_state=run_config["random_state"],
    )

    return model


def get_current_model(
    models: List[DriftModel],
    i,
    t,
    model_type: str,
    last_model_used_idx=None,
):
    idx_to_add = 0
    if last_model_used_idx is not None:
        models = models[last_model_used_idx:]
        idx_to_add = last_model_used_idx

    model_idx = -1
    for model in models:
        if model_type == "ml":
            t_model = model.ml_available_time

        elif model_type == "drift":
            t_model = model.drift_available_time
        else:
            raise NotImplementedError

        i_model = model.end_idx
        if (t_model <= t) and (i_model <= i):
            model_idx += 1
        else:
            break

    model_idx = model_idx + idx_to_add
    model_idx = max(0, model_idx)
    return model_idx


def get_current_models(
    models: List[DriftModel],
    i,
    t,
    last_ml_model_used=None,
    last_drift_model_used=None,
) -> Tuple[int, int]:
    ml_model_idx, drift_model_idx = (
        get_current_model(models, i, t, "ml", last_ml_model_used),
        get_current_model(models, i, t, "drift", last_drift_model_used),
    )
    if models[drift_model_idx].drift_detector.needs_model():
        assert ml_model_idx == drift_model_idx
    return ml_model_idx, drift_model_idx


def compute_y_scores(
    model: Model,
    current_index: int,
    current_model_i: int,
    model_used: np.ndarray,
    y_scores: np.ndarray,
    x: Union[np.ndarray, pd.DataFrame],
    predict_forward: int,
    last_idx: int,
):
    logger = logging.getLogger(__name__)
    if model_used[current_index] < current_model_i:
        logger.debug(f"Seeing forward at index {current_index}")
        end_idx = min(current_index + predict_forward, last_idx)

        if isinstance(model, LazyPipeline):
            y_scores[current_index:end_idx] = model.lazy_predict(
                current_index, end_idx
            )
        else:
            x_to_pred = x[current_index:end_idx]
            if model.objective in ["regression"]:
                y_pred = model.predict(x_to_pred)
            elif model.objective in ["binary", "classification"]:
                y_pred = model.predict_proba(x_to_pred)
            else:
                raise NotImplementedError
            y_scores[current_index:end_idx] = y_pred

        model_used[current_index:end_idx] = current_model_i
    return y_scores, model_used


def get_ref_eval_config(configs: dict, ref_config_names: List[str]):
    ref_configs = []
    eval_configs = []
    for config in configs.get("runs"):
        if config.get("name") in ref_config_names:
            ref_configs.append(config)
        else:
            eval_configs.append(config)
    return ref_configs, eval_configs


def get_common_detectors_params(
    config: dict, metadata: pd.DataFrame, start_idx: int, end_idx: int
):
    auto_detector_params = {
        "x_metadata": metadata,
        "features": metadata["feature"].to_list(),
        "numerical_features": metadata["feature"][
            metadata["type"] != "cat"
        ].to_list(),
        "categorical_features": metadata["feature"][
            metadata["type"] == "cat"
        ].to_list(),
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    return {**config.get("common_detectors_params"), **auto_detector_params}


def quite_model(model: Model):
    l_model = model
    if isinstance(l_model, Pipeline):
        l_model = l_model[-1]
    if isinstance(l_model, SkModel):
        l_model.model.set_params(**{"verbose": 0})


def add_model(
    models: List[DriftModel],
    model_path: str,
    f_new_model: Callable[[], Model],
    f_new_detector: Callable[[], DriftDetector],
    x_idx,
    delays: Delays,
    x,
    y,
    t,
    start_idx,
    end_idx,
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:
    model = f_new_model()
    model = load_do_save_model(
        model,
        model_path,
        x.iloc[start_idx:end_idx],
        y[start_idx:end_idx],
        lock_model_writing,
        list_model_writing,
    )
    quite_model(model)

    if isinstance(model, LazyPipeline):
        y_scores = model.safe_lazy_predict(x, start_idx, end_idx)
    else:
        if model.objective in ["regression"]:
            y_scores = model.predict(x.iloc[start_idx:end_idx])
        elif model.objective in ["binary", "classification"]:
            y_scores = model.predict_proba(x.iloc[start_idx:end_idx])
        else:
            raise NotImplementedError

    drift_detector = f_new_detector(start_idx, end_idx)
    drift_detector.fit(
        x=x.iloc[start_idx:end_idx],
        t=t[start_idx:end_idx],
        y=y[start_idx:end_idx],
        y_scores=y_scores,
        model=model,
    )

    models.append(
        DriftModel(
            t[x_idx] + delays.ml_model,
            model,
            t[x_idx] + delays.drift_detector,
            drift_detector,
            0,
            end_idx,
        )
    )


def free_mem_models(
    models: List[DriftModel], ml_model_idx: int, drift_model_idx: int
) -> None:
    for i in range(len(models)):
        if i < ml_model_idx:
            models[i].ml_model = None
        if i < drift_model_idx:
            models[i].drift_detector = None
