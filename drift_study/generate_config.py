import copy
import logging
import os
from typing import Any, Dict, Iterable, List, Tuple, Union

import configutils
import yaml

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def gen_dict_extract(
    key: str, var: Any
) -> Iterable[Tuple[List[Union[str, int]], Any]]:
    if hasattr(var, "items"):
        for k, v in var.items():
            if k == key:
                yield [], v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield (
                        [
                            k,
                        ]
                        + result[0],
                        result[1],
                    )
            elif isinstance(v, list):
                for i, d in enumerate(v):
                    for result in gen_dict_extract(key, d):
                        yield (
                            [
                                k,
                                i,
                            ]
                            + result[0],
                            result[1],
                        )


def value_to_name(value: Any) -> str:
    if isinstance(value, dict):
        if "retraining" in value:
            return str(value["retraining"])
    return str(value)


def generate_from_param(
    runs: List[Dict[str, Any]], key: List[Union[int, str]], values: List[Any]
) -> List[Dict[str, Any]]:
    out_runs = []
    for run in runs:
        for v in values:
            new_run = copy.deepcopy(run)
            deep_update(new_run, key, copy.deepcopy(v))
            new_run["name"] = new_run["name"] + "_" + value_to_name(v)
            out_runs.append(new_run)
    return out_runs


def deep_update(
    config: Union[List[Any], Dict[str, Any]],
    key: List[Union[str, int]],
    value: Any,
) -> Dict[str, Any]:
    if len(key) == 0:
        return value

    current_key = key[0]
    if isinstance(config, List) and not isinstance(current_key, int):
        raise ValueError
    if isinstance(config, Dict) and not isinstance(current_key, str):
        logger.error(config, str)
        raise ValueError

    config[current_key] = deep_update(config[current_key], key[1:], value)
    return config


def run(config: Dict[str, Any]) -> None:

    output_path = config.pop("output_path")
    runs = config["runs"]
    all_runs: List[Dict[str, Any]] = []
    for r in runs:
        out_runs = [copy.deepcopy(r)]
        for key, values in gen_dict_extract("grid", r):
            out_runs = generate_from_param(out_runs, key, values)

        all_runs = all_runs + out_runs

    out_config = copy.deepcopy(config)
    out_config["runs"] = all_runs
    with open(output_path, "w") as f:
        yaml.dump(out_config, f)

    logger.info(f"{len(all_runs)} config generated.")


if __name__ == "__main__":
    config = configutils.get_config()
    run(config)
