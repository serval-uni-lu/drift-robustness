import logging
import os

import configutils
import h5py
import joblib
import numpy as np
import pandas as pd
from mlc.datasets.dataset_factory import get_dataset
from sklearn.metrics import f1_score, precision_score, recall_score

from drift_study.utils.helpers import get_ref_eval_config

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run():
    config = configutils.get_config()
    print(config)
    ref_configs, eval_configs = get_ref_eval_config(
        config, config.get("evaluation_params").get("reference_methods")
    )

    dataset = get_dataset(config.get("dataset"))
    model_name = config.get("runs")[0].get("model").get("name")
    logger.info(f"Starting dataset {dataset.name}, model {model_name}")
    x, y, t = dataset.get_x_y_t()

    test_i = np.arange(len(x))[config.get("window_size") :]

    batch_size = config.get("batch_size")
    length = len(test_i) - (len(test_i) % batch_size)
    index_batches = np.split(test_i[:length], length / batch_size)

    for run_config in config.get("runs"):
        drift_data_path = (
            f"./data/{dataset.name}/drift/"
            f"{run_config.get('model').get('name')}_{run_config.get('name')}"
        )
        with h5py.File(drift_data_path, "r") as f:
            y_scores = f["y_scores"][()]
            model_used = f["model_used"][()]

        y_pred = np.argmax(y_scores, axis=1)

        run_config["is_retrained"] = []
        for i, index_batch in enumerate(index_batches):
            if i == 0:
                run_config["is_retrained"].append(True)
            else:
                if (
                    model_used[index_batch].max()
                    != model_used[index_batches[i - 1]].max()
                ):
                    run_config["is_retrained"].append(True)
                else:
                    run_config["is_retrained"].append(False)

        run_config["f1s"] = [
            f1_score(y[index_batch], y_pred[index_batch])
            for index_batch in index_batches
        ]
        run_config["model_used"] = model_used

    significance = config.get("evaluation_params").get("significance")
    out = []
    for ref_config in ref_configs:
        ref_config_name = ref_config.get("drift_name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_f1s = ref_config.get("f1s")

        for eval_config in config.get("runs"):
            TP, TN, FP, FN = 0, 0, 0, 0
            drift_pred = np.full(len(index_batches), np.nan)
            drift_true = np.full(len(index_batches), np.nan)
            eval_config_name = eval_config.get("name")
            logger.info(
                f"<><><> <><><> Config test: {eval_config_name} <><><> <><><>"
            )

            model_used = eval_config.get("model_used")
            drift_data_path = (
                f"./data/{dataset.name}/drift/"
                f"{model_name}_{eval_config.get('name')}"
            )
            with h5py.File(drift_data_path, "r") as f:
                model_start_indexes = f["model_start_indexes"][()]
                model_end_indexes = f["model_end_indexes"][()]

            model_path = f"./models/{dataset.name}/{model_name}"
            fitted_models = [
                joblib.load(
                    f"{model_path}_{model_start_indexes[i]}_"
                    f"{model_end_indexes[i]}.joblib"
                )
                for i in np.arange(np.max(model_used))
            ]
            retraineds = eval_config.get("is_retrained")
            config_f1 = eval_config.get("f1s")

            for i, index_batch in enumerate(index_batches):
                if retraineds[i]:
                    if i > 0:
                        y_scores = fitted_models[
                            model_used[index_batch].max() - 1
                        ].predict_proba(x.iloc[index_batch])
                        y_pred = np.argmax(y_scores, axis=1)
                        f1_past = f1_score(y[index_batch], y_pred)
                        if (f1_past + significance) <= config_f1[i]:
                            TP += 1
                            drift_true[i] = 1
                            drift_pred[i] = 1
                        else:
                            FP += 1
                            drift_true[i] = 0
                            drift_pred[i] = 1

                else:
                    if ref_f1s[i] <= (config_f1[i] + significance):
                        TN += 1
                        drift_true[i] = 0
                        drift_pred[i] = 0
                    else:
                        FN += 1
                        drift_true[i] = 1
                        drift_pred[i] = 0

            precision = precision_score(drift_true[1:], drift_pred[1:])
            recall = recall_score(drift_true[1:], drift_pred[1:])
            logger.info(f"Precision: {precision}, Recall:  {recall}")
            out.append(
                {
                    "reference": ref_config.get("name"),
                    "eval": eval_config.get("name"),
                    "n_train": eval_config.get("model_used").max() + 1,
                    "tn": TN,
                    "fp": FP,
                    "fn": FN,
                    "tp": TP,
                    "precision": precision,
                    "recall": recall,
                }
            )
    out_path = f"./reports/{dataset.name}/{model_name}_confusion.csv"
    out = pd.DataFrame(out)
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    run()
