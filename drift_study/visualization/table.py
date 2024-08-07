import copy
import logging.config
from typing import Any, Dict, List

import configutils
import numpy as np
import pandas as pd
from configutils.utils import merge_parameters
from mlc.metrics.metric import Metric
from mlc.metrics.metric_factory import create_metric

from drift_study.utils.evaluation import load_config_eval
from drift_study.utils.helpers import initialize
from drift_study.utils.logging import configure_logger
from mlc.load_do_save import load_json
from pathlib import Path


BIG_NUMBER = 1_000_000_000

def process_result(res):
    conf = res["config"]
    runs = conf["runs"]
    out = {
        "model": runs["model"]["name"],
        "delay": f"{runs['delays']['label']}_{runs['delays']['retraining']}",
        "period": runs["detectors"][0]["params"]["period"],
        "window_size": runs["train_window_size"],
        "n_train": res["n_train"],
        "ml_metric": res["ml_metric"]
    }
    return out

def process_results(results):
    return [process_result(res) for res in results]
        
        
def find_conf_file(sub_dir_path, dataset_name, run):
    model_name = run["model"]["name"]
    run_name = run["name"]
    return f"./data/optimizer_results/{dataset_name}/{model_name}/{sub_dir_path}/{run_name}.json"

def run(
    config: Dict[str, Any],
) -> None:
    configure_logger(config)
    logger = logging.getLogger(__name__)

    
    dataset_name = config["dataset"]["name"]
    sub_dir_path = config["sub_dir_path"]
    
    results = []
    for i in range(len(config.get("runs"))):
        result = load_json(find_conf_file(sub_dir_path, dataset_name, config["runs"][i]))
        results.append(result)   
    logger.info(f"{len(results)} results loaded.")

    results = process_results(results)
    logger.info(f"{len(results)} results processed.")
    
    results  = pd.DataFrame(results)
    out_path = f"./reports/{dataset_name}/{sub_dir_path}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logger.info("Table saved.")


if __name__ == "__main__":
    config_all = configutils.get_config()
    configure_logger(config_all)
    run(config_all)
