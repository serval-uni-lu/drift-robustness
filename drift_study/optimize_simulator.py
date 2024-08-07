import copy
import logging.config
import os
from multiprocessing import Manager
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import configutils
import joblib
import numpy as np
import numpy.typing as npt
import optuna
from configutils.utils import merge_parameters
from joblib import Parallel, delayed, parallel_backend
from optuna import Study
from optuna._callbacks import MaxTrialsCallback, RetryFailedTrialCallback
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState
from sklearn.model_selection import TimeSeriesSplit

from drift_study import run_simulator
from drift_study.drift_detectors.drift_detector_factory import (
    get_drift_detector_class_from_conf,
)
from drift_study.utils.io_utils import manual_save_run
from drift_study.utils.logging import configure_logger


def update_params(
    trial: optuna.Trial,
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
) -> None:
    logger = logging.getLogger(__name__)
    run_config["name"] = run_config["name"] + str(trial.number)
    for i, e in enumerate(list_drift_detector):
        new_params = merge_parameters(
            e["params"],
            e["detector"].define_trial_parameters(
                trial, trial_params=config["trial_params"]
            ),
        )
        logger.debug(new_params)
        run_config["detectors"][i]["params"] = new_params

    config["runs"] = [run_config]


def get_default_params(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out = {}
    for i, e in enumerate(list_drift_detector):
        out = merge_parameters(
            out,
            e["detector"].get_default_params(
                trial_params=config["trial_params"]
            ),
        )
    return out


def execute_one_fold(
    fold_idx: int,
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    train_idx: npt.NDArray[np.int_],
    test_idx: npt.NDArray[np.int_],
    lock_model_writing: Optional[LockType] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float]:

    run_config["name"] = run_config["name"] + f"_f{fold_idx}"
    logger = logging.getLogger(__name__)
    logger.info(f"Starting { run_config['name'] }...")

    if run_config.get("use_all_test", True):
        run_config["last_idx"] = run_config["test_start_idx"]
    else:
        run_config["last_idx"] = int(test_idx[-1])

    run_config["test_start_idx"] = int(test_idx[0])

    print(run_config["last_idx"])
    # run_config["n_early_stopping"] = floor(
    #     (test_idx[-1] - train_idx[-1])
    #     / config["trial_params"]["period"]["min"]
    # )
    config["runs"] = [run_config]
    n_train, ml_metric = run_simulator.run(
        config, 0, lock_model_writing, list_model_writing, verbose=1
    )
    logger.info(f"Completed {run_config['name']}.")
    return n_train, ml_metric


def execute_one_trial(
    trial: optuna.Trial,
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
    lock_model_writing: Optional[LockType] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    configure_logger(config)

    update_params(
        trial,
        config,
        run_config,
        list_drift_detector,
    )

    tscv = TimeSeriesSplit(n_splits=config["evaluation_params"]["n_splits"])

    test_start_idx = run_config["test_start_idx"]
    metrics = []
    for i, (train_index, test_index) in reversed(
        list(enumerate(tscv.split(np.arange(test_start_idx))))
    ):
        metrics.append(
            execute_one_fold(
                i,
                copy.deepcopy(config),
                copy.deepcopy(run_config),
                train_index,
                test_index,
                lock_model_writing,
                list_model_writing,
            )
        )

    n_train = [m[0] for m in metrics]
    ml_metric = [m[1] for m in metrics]

    manual_save_run(config, run_config, n_train, ml_metric)

    return float(np.mean(n_train)), float(np.mean(ml_metric))


def run(
    config: Dict[str, Any],
    run_i: int,
    lock_model_writing: Optional[LockType] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:
    configure_logger(config)
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    logger = logging.getLogger(__name__)
    # CONFIG
    run_config = merge_parameters(
        config.get("common_runs_params"), config["runs"][run_i]
    )
    logger.info(f"Optimizing config {run_config.get('name')}")

    # LOAD AND CREATE OBJECTS
    list_drift_detector = get_drift_detector_class_from_conf(
        run_config.get("detectors")
    )

    study_name = run_config["name"]
    model_name = run_config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    sub_dir_path = config["sub_dir_path"]

    studies_dir = (
        f"./data/optimizer/{dataset_name}/{model_name}/{sub_dir_path}/"
    )

    studies_path = f"{studies_dir}/study_{study_name}.db"
    Path(studies_path).parent.mkdir(parents=True, exist_ok=True)

    # warnings.filterwarnings("ignore", category=ExperimentalWarning)

    failed_trial_callback = RetryFailedTrialCallback(max_retry=None)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{studies_path}",
        heartbeat_interval=10,
        grace_period=20,
        failed_trial_callback=failed_trial_callback,
    )

    sampler_path = f"{studies_dir}/study_{study_name}.sampler"
    if os.path.exists(sampler_path):
        sampler = joblib.load(sampler_path)
    else:
        sampler = TPESampler(n_startup_trials=5, seed=42, multivariate=True)
        joblib.dump(
            sampler,
            sampler_path,
        )

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        storage=storage,
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )
    n_to_finish = config["trial_params"]["n_trials"]
    if run_config["detectors"][0]["name"] == "no_detection":
        n_to_finish = 1

    n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
    logger.info(f"Completed {study_name}: {n_completed}")

    def logger_done_callback(
        study_l: Study, frozen_trial: FrozenTrial
    ) -> None:
        n_done = len(study_l.get_trials(states=(TrialState.COMPLETE,)))
        logger.info(f"Completed {study_name}: {n_done}")

    if n_completed == 0:
        default_params = get_default_params(
            config, run_config, list_drift_detector
        )
        if default_params is not None:
            study.enqueue_trial(default_params)

    if n_completed < n_to_finish:
        study.optimize(
            lambda trial_l: execute_one_trial(
                trial_l,
                copy.deepcopy(config),
                copy.deepcopy(run_config),
                list_drift_detector,
                lock_model_writing,
                list_model_writing,
            ),
            callbacks=[
                MaxTrialsCallback(
                    n_to_finish,
                    states=(TrialState.COMPLETE,),
                ),
                lambda *_: joblib.dump(sampler, sampler_path),
                logger_done_callback,
            ],
        )


def run_many(config: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    n_jobs_optimiser = (
        config_all["performance"].get("n_jobs", {}).get("optimizer", 1)
    )

    if n_jobs_optimiser == 1:
        logger.info("Running in sequence.")
        for i in range(len(config_all.get("runs"))):
            run(copy.deepcopy(config_all), i)
    else:
        logger.info("Running in parallel.")
        with Manager() as manager:
            lock = manager.Lock()
            dico: Dict[str, Any] = manager.dict()
            with parallel_backend("loky", n_jobs=n_jobs_optimiser):
                Parallel()(
                    delayed(run)(copy.deepcopy(config_all), i, lock, dico)
                    for i in range(len(config_all.get("runs")))
                )


if __name__ == "__main__":
    config_all = configutils.get_config()
    configure_logger(config_all)
    run_many(config_all)
