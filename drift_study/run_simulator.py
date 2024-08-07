import copy
import logging
import math
import os
from multiprocessing import Lock, Manager
from typing import Any, Dict, List, Optional, Tuple, Union

import configutils
import numpy as np
from configutils.utils import merge_parameters
from joblib import Parallel, delayed, parallel_backend
from mlc.metrics.metric_factory import create_metric
from mlc.metrics.metrics import PredClassificationMetric
from tqdm import tqdm

from drift_study.utils.delays import get_delays
from drift_study.utils.drift_model import DriftModel
from drift_study.utils.helpers import (
    add_model,
    compute_y_scores,
    free_mem_models,
    get_current_models,
    initialize,
)
from drift_study.utils.io_utils import save_drift_run
from drift_study.utils.logging import configure_logger


def round_up_to_multiple(a, n):
    return int(math.ceil(a / n)) * n


def get_start_end_idx(end_idx: int, windows_size: int) -> Tuple[int, int]:
    if windows_size > 0:
        start_idx = max(end_idx - windows_size, 0)
    else:
        start_idx = 0
    return start_idx, end_idx


def run(
    config,
    run_i,
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
    verbose=1,
) -> Tuple[int, Union[float, List[float]]]:
    # CONFIG
    configure_logger(config)
    logger = logging.getLogger(__name__)
    run_config = merge_parameters(
        config.get("common_runs_params"), config.get("runs")[run_i]
    )
    logger.info(f"Running config {run_config.get('name')}")
    model_root_dir = config.get(
        "models_dir", os.environ.get("MODELS_DIR", "./models")
    )

    # INITIALIZE PARAMETERS
    test_start_idx = run_config["test_start_idx"]
    train_window_size = run_config["train_window_size"]
    first_train_window_size = run_config["first_train_window_size"]
    training_step_size = run_config.get("training_step_size", 1)

    predict_forward = config.get("performance").get("predict_forward")
    n_early_stopping = run_config.get("n_early_stopping", 0)
    early_stopped = False

    # LOAD AND CREATE OBJECTS
    dataset, f_new_model, f_new_detector, x, y, t = initialize(
        config, run_config
    )

    logger.debug(f"N inputs {len(y)}, with distribution {y.mean()}.")

    delays = get_delays(run_config, f_new_detector(0, 0))
    last_idx = run_config.get("last_idx", -1)
    if last_idx == -1:
        last_idx = len(x)

    # PREPARE DATASTRUCTURES
    models: List[DriftModel] = []
    model_used = np.full(last_idx, -1)
    if config["evaluation_params"]["n_score"] == 1:
        y_scores = np.full(last_idx, np.nan)
    else:
        y_scores = np.full(
            (last_idx, config["evaluation_params"]["n_score"]), np.nan
        )
    is_drifts = np.full(last_idx, np.nan)
    is_drift_warnings = np.full(last_idx, np.nan)
    last_ml_model_used = 0
    last_drift_model_used = 0
    metrics = []
    model_name = f_new_model().name

    if run_config["random_state"] != 42:
        model_name = f"{model_name}_{str(run_config['random_state'])}"

    # TRAIN FIRST MODEL

    drift_data_path = (
        f"./data/simulator/"
        f"{config['dataset']['name']}/{model_name}/"
        f"{config['sub_dir_path']}/{run_config['name']}.hdf5"
    )
    logger.debug(drift_data_path)
    if os.path.exists(drift_data_path):
        logger.info("Path exists, skipping.")
        return -1, -1

    start_idx, end_idx = get_start_end_idx(
        test_start_idx, first_train_window_size
    )
    model_path = (
        f"{model_root_dir}/{dataset.name}/"
        f"{model_name}_{start_idx}_{end_idx}.joblib"
    )

    logger.debug(f"start_index {start_idx}, end_index {end_idx}.")

    add_model(
        models,
        model_path,
        f_new_model,
        f_new_detector,
        0,
        delays,
        x,
        y,
        t,
        start_idx,
        end_idx,
        lock_model_writing,
        list_model_writing,
    )

    # Main loop
    for x_idx in tqdm(
        np.arange(test_start_idx, last_idx), disable=(verbose == 0)
    ):

        # Find current model
        ml_model_idx, drift_model_idx = get_current_models(
            models, x_idx, t[x_idx], last_ml_model_used, last_drift_model_used
        )
        last_ml_model_used, last_drift_model_used = (
            ml_model_idx,
            drift_model_idx,
        )
        # logger.debug(ml_model_idx)
        model = models[ml_model_idx].ml_model
        drift_detector = models[drift_model_idx].drift_detector

        # Update predictions if needed
        y_scores, model_used = compute_y_scores(
            model,
            x_idx,
            ml_model_idx,
            model_used,
            y_scores,
            x,
            predict_forward,
            last_idx,
        )
        if not early_stopped:
            # Detect drift
            (
                is_drifts[x_idx],
                is_drift_warnings[x_idx],
                metric,
            ) = drift_detector.update(
                x=x.iloc[x_idx : x_idx + 1],
                t=t[x_idx : x_idx + 1],
                y=y[x_idx : x_idx + 1],
                y_scores=y_scores[x_idx : x_idx + 1],
            )
            metrics.append(metric)
            # Do not retrain if we are
            # not using the latest drift model available
            # Due to delay
            if drift_model_idx == len(models) - 1:
                end_idx = round_up_to_multiple(x_idx + 1, training_step_size)
                if is_drifts[x_idx] and (end_idx < last_idx):
                    logger.debug(f"Drift at index {x_idx}")
                    start_idx, end_idx = get_start_end_idx(
                        end_idx, train_window_size
                    )

                    logger.debug(
                        f"start_index {start_idx}, end_index {end_idx}."
                    )

                    model_path = (
                        f"{model_root_dir}/{dataset.name}/"
                        f"{model_name}_{start_idx}_{end_idx}.joblib"
                    )
                    add_model(
                        models,
                        model_path,
                        f_new_model,
                        f_new_detector,
                        round_up_to_multiple(x_idx, training_step_size),
                        delays,
                        x,
                        y,
                        t,
                        start_idx,
                        end_idx,
                        lock_model_writing,
                        list_model_writing,
                    )
                    if len(models) > 1:
                        assert (
                            models[-1].drift_detector
                            != models[-2].drift_detector
                        )
                        assert models[-1].ml_model != models[-2].ml_model

                    # Avoid memory full
                    free_mem_models(models, ml_model_idx, drift_model_idx)
                    # Early stop
                    if (0 < n_early_stopping) and (
                        n_early_stopping <= len(models)
                    ):
                        early_stopped = True

    # Metrics
    n_train = int(np.max(model_used) + 1)
    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    if isinstance(prediction_metric, PredClassificationMetric):
        y_scores = np.argmax(y_scores, axis=1)

    val_test_idx = config["evaluation_params"].get("val_test_idx")
    if val_test_idx is None:
        metric = prediction_metric.compute(
            y[test_start_idx:last_idx], y_scores[test_start_idx:last_idx]
        )

    else:
        metric_idxs = [
            (test_start_idx, last_idx),
            (test_start_idx, val_test_idx),
            (val_test_idx, last_idx),
        ]
        metric = [
            prediction_metric.compute(
                y[m_idx_start:m_idx_end], y_scores[m_idx_start:m_idx_end]
            )
            for m_idx_start, m_idx_end in metric_idxs
        ]

    # Save
    save_drift_run(
        numpy_to_save={
            "is_drifts": is_drifts,
            "is_drift_warnings": is_drift_warnings,
            "y_scores": y_scores,
            "model_used": model_used,
        },
        models=models,
        metrics=metrics,
        dataset_name=dataset.name,
        model_name=model_name,
        run_name=run_config.get("name"),
        sub_dir_path=config["sub_dir_path"],
        config=copy.deepcopy(config),
        run_config=copy.deepcopy(run_config),
        n_train=n_train,
        ml_metric=metric,
    )

    return n_train, metric


def run_many(
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:
    config_all = configutils.get_config()

    run_idx = config_all.get("run_idx")
    if run_idx is not None:
        config_all["runs"] = [config_all["runs"][run_idx]]

    n_jobs = config_all["performance"].get("n_jobs", {}).get("simulator", 1)
    if lock_model_writing is None:
        with Manager() as manager:
            lock = manager.Lock()
            dico = manager.dict()
            with parallel_backend("loky", n_jobs=n_jobs):
                Parallel()(
                    delayed(run)(copy.deepcopy(config_all), i, lock, dico)
                    for i in range(len(config_all.get("runs")))
                )
    else:
        with parallel_backend("loky", n_jobs=n_jobs):
            Parallel()(
                delayed(run)(
                    copy.deepcopy(config_all),
                    i,
                    lock_model_writing,
                    list_model_writing,
                )
                for i in range(len(config_all.get("runs")))
            )


if __name__ == "__main__":
    run_many()

# write function to round up a to a multiple of n
