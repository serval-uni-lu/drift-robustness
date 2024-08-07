import glob
import os
from typing import Any, Dict, List

import configutils
from mlc.load_do_save import load_json, save_json
from tqdm import tqdm


def load_confs(input_path: str) -> List[Dict[str, Any]]:

    if os.path.isdir(input_path):
        out = []
        list_files = glob.glob(f"{input_path}/*.json")
        for path in tqdm(list_files, total=len(list_files)):
            out.append(load_json(path))
        return out

    if os.path.isfile(input_path):
        return load_json(input_path)

    if os.path.isfile(f"{input_path}.json"):
        return load_json(f"{input_path}.json")

    raise NotImplementedError


def run(
    input_dir: str,
) -> None:
    config = configutils.get_config()
    print(config)

    input_dir = input_dir

    conf_results = load_confs(f"{input_dir}")

    out = f"{input_dir}.json"

    save_json(conf_results, out)


def merge_date(input_dir: str):

    original = load_confs(f"./rf_complete/{input_dir}")
    dates = load_confs(f"./dates/{input_dir}")
    dates_key = {e["id"]: e for e in dates}
    out = f"./merged/{input_dir}.json"
    for conf in original:
        date = dates_key[conf["config"]["runs"][0]["name"]]["n_train"]
        assert date[0] == conf["n_train"]
        conf["n_train"] = date

    save_json(original, out)


def run_many() -> None:

    DELAYS = [
        "all_delays",
        "all_delays_half",
        "all_delays_twice",
        "no_delays",
    ]

    for delay in DELAYS:
        merge_date(f"{delay}")


if __name__ == "__main__":
    run_many()
