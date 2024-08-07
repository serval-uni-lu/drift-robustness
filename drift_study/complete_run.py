import copy
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import configutils
import numpy as np
import numpy.typing as npt
import yaml
from mlc.load_do_save import load_json

from drift_study.utils.pareto import calc_pareto_rank

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def load_config_from_dir(input_dir: str) -> List[Dict[str, Any]]:
    json_files = [
        path for path in os.listdir(input_dir) if (path.endswith(".json") and (not(any([path.endswith(f"f{e}.json") for e in range(4)]))))
    ]

    optimizer_results = [
        load_json(f"{input_dir}/{path}") for path in json_files
    ]

    return optimizer_results


def get_run_name(config: Dict[str, Any]) -> str:
    if "config" in config:
        try:
            run_name = config["config"]["runs"][0]["name"]
        except KeyError:
            run_name = config["config"]["runs"]["name"]
    else:
        run_name = config["runs"][0]["name"]

    index = re.search(r"\d", run_name).start()
    return run_name[:index]


def get_metrics(config: Dict[str, Any]) -> Tuple[float, float]:
    n_train, ml_metric = config["n_train"], config["ml_metric"]
    if isinstance(n_train, list):
        n_train = float(np.mean(n_train))
        ml_metric = float(np.mean(ml_metric))

    return n_train, ml_metric


def group_by_name(
    configs: List[Dict[str, Any]]
) -> Dict[str, npt.NDArray[np.int_]]:
    str_list = [get_run_name(e) for e in configs]
    out: Dict[str, npt.NDArray[np.int_]] = {}

    for i, e in enumerate(str_list):
        if e in out:
            out[e] = np.append(out[e], i)
        else:
            out[e] = np.array([i])

    return out


def filter_n_train(result: Dict[str, Any]):
    # No detection must be run for baseline
    if get_run_name(result) == "no_detection":
        return True

    # We do not execute the run if it never triggered retrain for any fold.
    return get_metrics(result)[0] > 1.0


def pareto_rank_by_group(
    optimize_configs: List[Dict[str, Any]],
    grouped: bool = True,
) -> npt.NDArray[np.int_]:
    configs_group = group_by_name(optimize_configs)
    configs_rank_in_group = np.full(len(optimize_configs), -1)
    objective_direction = np.array([1, -1])

    config_metrics = np.array([list(get_metrics(e)) for e in optimize_configs])

    if not grouped:
        return calc_pareto_rank(config_metrics, objective_direction)

    for group in configs_group.keys():
        group_idxs = configs_group[group]
        if group == "no_detection":
            # We want to run a single no detection, therefore
            # Pareto rank 1 for first no detection, infinity for the others.
            pareto_rank = np.array(
                [1]
                + [np.iinfo(np.int32).max]
                * (len(configs_rank_in_group[configs_group[group]]) - 1)
            )
        else:
            pareto_rank = calc_pareto_rank(
                config_metrics[group_idxs],
                objective_direction,
            )
        configs_rank_in_group[group_idxs] = pareto_rank
        print(
            f"Detector {group}, {(pareto_rank == 1).sum()} / {len(group_idxs)}"
        )

    return configs_rank_in_group


def filter_configs(
    config: Dict[str, Any], optimize_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    # Filter by n_train (n_train > 1 for all detectors except no detection)
    optimize_configs = list(
        filter(filter_n_train, copy.deepcopy(optimize_configs))
    )

    # Evaluate
    configs_rank_in_group = pareto_rank_by_group(optimize_configs)

    # Filter by max config
    pareto_filter = configs_rank_in_group <= int(config["max_pareto"])
    optimize_configs = [
        e for e, f in zip(optimize_configs, pareto_filter) if f
    ]

    return optimize_configs


def update_config_to_run(
    config: Dict[str, Any], optimize_config: Dict[str, Any]
) -> Dict[str, Any]:

    # Get config
    config_l = optimize_config["config"]
    try:
        config_run = config_l["runs"][0]
    except KeyError:
        config_run = config_l["runs"]

    # Update
    config_run["last_idx"] = -1
    config_l["n_early_stopping"] = config.get("n_early_stopping", -1)

    retraining_delay = config.get("retraining_delay")
    if retraining_delay is not None:
        config_l["common_runs_params"]["delays"][
            "retraining"
        ] = retraining_delay
        config_run["delays"]["retraining"] = retraining_delay

    return config_l


def short_config(config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    out = []
    for e in config:
        try:
            r = e["runs"][0]
        except KeyError:
            r = e["runs"]
        out.append(
            {
                "name": r["name"],
                "type": r["type"],
                "detectors": r["detectors"],
            }
        )
    return out


def run() -> None:
    config = configutils.get_config()

    optimize_configs = load_config_from_dir(config["input_dir"])
    configs_to_run = filter_configs(config, copy.deepcopy(optimize_configs))
    configs_to_run = [
        update_config_to_run(config, copy.deepcopy(e)) for e in configs_to_run
    ]
    logger.info(
        f"That would run {len(configs_to_run)} out of {len(optimize_configs)}"
    )

    output_file = config.get("output_file")
    if config.get("output_file"):
        logger.info(f"Saving configs to {output_file}.")
        out = {
            "runs": short_config(configs_to_run),
        }
        with open(output_file, "w") as f:
            yaml.dump(out, f)


if __name__ == "__main__":
    run()
