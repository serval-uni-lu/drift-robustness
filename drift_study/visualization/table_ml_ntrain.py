from pathlib import Path

import configutils
import numpy as np
import pandas as pd

from drift_study.reports.naming import beautify_dataframe
from drift_study.reports.result_processor import filter_conf, load_confs
from drift_study.utils.pareto import calc_pareto_rank

BIG_RANK = 1000000


def filter_extreme(n_train, pareto_rank, extreme_value: int):
    for i in range(len(n_train)):
        if n_train[i] > extreme_value:
            pareto_rank[i] = BIG_RANK
    return pareto_rank


def run() -> None:
    config = configutils.get_config()
    print(config)

    input_dir = config["input_dir"]
    output_file = config.get("output_file", None)

    conf_results = load_confs(input_dir)
    conf_results = [filter_conf(e) for e in conf_results]

    df = pd.DataFrame(conf_results)

    # Add rank
    pareto_rank = calc_pareto_rank(
        np.array([df["n_train"], df["ml_metric"]]).T, np.array([1, -1])
    )
    max_n_train = config.get("max_n_train", -1)
    if max_n_train > 0:
        pareto_rank = filter_extreme(
            df["n_train"].to_numpy(), pareto_rank, max_n_train
        )

    df["pareto_rank"] = pareto_rank

    df = beautify_dataframe(df, add_missing=True)

    df = df.drop_duplicates(subset="detector_name", ignore_index=True)

    df = df[["Detector", "Type", "Pareto / Total"]]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(df)


if __name__ == "__main__":
    run()
