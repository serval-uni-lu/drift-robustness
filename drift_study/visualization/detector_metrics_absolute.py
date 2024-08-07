import logging
import os

import configutils
import numpy as np
import pandas as pd
from configutils.utils import merge_parameters
from mlc.datasets.dataset_factory import get_dataset
from mlc.metrics.metric_factory import create_metric

from drift_study.utils.evaluation import load_config_eval

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array,
        indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def calc_pareto_rank(df: pd.DataFrame) -> pd.DataFrame:

    costs = df[["n_train", "metric"]].to_numpy()
    costs[:, 1] = np.max(costs[:, 1]) - costs[:, 1]
    costs, inverse, count = np.unique(
        costs, return_inverse=True, return_counts=True, axis=0
    )
    n = len(costs)

    to_front_idx = np.arange(n)
    rank = np.full(n, 1e16, dtype=int)
    i = 1
    while len(to_front_idx) > 0:
        current_cost = costs[to_front_idx]
        efficient = is_pareto_efficient_simple(current_cost)
        rank[to_front_idx[efficient]] = i
        to_front_idx = to_front_idx[~efficient]
        i += 1

    rank = rank[inverse]
    df["pareto_front"] = rank
    return df


def run():
    config = configutils.get_config()

    dataset = get_dataset(config.get("dataset"))
    for i in range(len(config.get("runs"))):
        config.get("runs")[i] = merge_parameters(
            config.get("common_runs_params").copy(),
            config.get("runs")[i].copy(),
        )

    model_name = config.get("runs")[0].get("model").get("name")
    logger.info(f"Starting dataset {dataset.name}, model {model_name}")
    x, y, t = dataset.get_x_y_t()

    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    config = load_config_eval(
        config, dataset, model_name, prediction_metric, y
    )
    out = []

    for eval_config in config.get("runs"):
        eval_config_name = eval_config.get("name")
        logger.info(f"Eval: {eval_config_name}")

        eval_metric = eval_config.get("prediction_metric")
        eval_metric_batch = eval_config.get("prediction_metric_batch")
        out.append(
            {
                "type": eval_config.get("type"),
                "method_name": eval_config_name,
                "n_train": eval_config.get("model_used").max() + 1,
                "metric": eval_metric,
                "metric_batch_min": np.min(eval_metric_batch),
                "metric_batch_mean": np.mean(eval_metric_batch),
                "metric_batch_max": np.max(eval_metric_batch),
            }
        )

    out_path = (
        f"./reports/{dataset.name}/"
        f"{model_name}_{prediction_metric.metric_name}_absolute.csv"
    )
    out = pd.DataFrame(out)
    out = calc_pareto_rank(out)
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    run()
