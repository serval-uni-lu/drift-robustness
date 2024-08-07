import logging
import os
from pathlib import Path

import configutils
import matplotlib
import numpy as np
from configutils.utils import merge_parameters
from matplotlib import pyplot as plt
from mlc.datasets.dataset_factory import get_dataset
from mlc.metrics.metric_factory import create_metric

from drift_study.utils.evaluation import load_config_eval
from drift_study.utils.helpers import get_ref_eval_config

matplotlib.use("TkAgg")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


markers = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^"]
font = {"size": 18}

matplotlib.rc("font", **font)


def run():
    config = configutils.get_config()
    print(config)

    for i in range(len(config.get("runs"))):
        config.get("runs")[i] = merge_parameters(
            config.get("common_runs_params"), config.get("runs")[i]
        )
    ref_configs, eval_configs = get_ref_eval_config(
        config, config.get("evaluation_params").get("reference_methods")
    )
    dataset = get_dataset(config.get("dataset"))
    model_name = config.get("runs")[0].get("model").get("name")
    logger.info(f"Starting dataset {dataset.name}, model {model_name}")
    x, y, t = dataset.get_x_y_t()

    batch_size = config.get("evaluation_params").get("batch_size")
    fig_folder = f"reports/{dataset.name}/"
    Path(fig_folder).mkdir(parents=True, exist_ok=True)

    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    # --- For each config, collect data needed

    config = load_config_eval(
        config, dataset, model_name, prediction_metric, y
    )
    for ref_config in ref_configs:
        ref_config_name = ref_config.get("name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        plt.figure(figsize=(20, 6))

        for i, eval_config in enumerate(config.get("runs")):
            eval_config_name = eval_config.get("name")
            model_used_max = eval_config.get("model_used").max()
            eval_metric = np.array(eval_config.get("prediction_metric_batch"))
            label = f"{eval_config_name}: {model_used_max + 1}"
            plt.plot(
                eval_metric, label=label, marker=markers[i % len(markers)]
            )

        plt.legend()
        plt.xlabel("Time ordered batch")
        plt.ylabel(prediction_metric.metric_name)
        Path(f"{fig_folder}/{ref_config.get('model').get('name')}/").mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(
            f"{fig_folder}/{ref_config.get('model').get('name')}/"
            f"{prediction_metric.metric_name}_"
            f"batch_{batch_size}_{ref_config_name}.pdf"
        )
        plt.clf()

    for i, ref_config in enumerate(ref_configs):
        ref_config_name = ref_config.get("name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_metric = np.array(ref_config.get("prediction_metric_batch"))
        plt.figure(figsize=(20, 6))
        for eval_config in config.get("runs"):
            eval_config_name = eval_config.get("name")
            model_used_max = eval_config.get("model_used").max()
            eval_metric = np.array(eval_config.get("prediction_metric_batch"))
            label = f"{eval_config_name}: {model_used_max + 1}"
            plt.plot(
                eval_metric - ref_metric,
                label=label,
                marker=markers[i % len(markers)],
            )

        plt.legend()
        plt.xlabel("Time ordered batch")
        plt.ylabel(prediction_metric.metric_name)
        plt.savefig(
            f"{fig_folder}/{ref_config.get('model').get('name')}/"
            f"{prediction_metric.metric_name}_"
            f"batch_{batch_size}_{ref_config_name}_delta.pdf"
        )
        plt.clf()


if __name__ == "__main__":
    run()
