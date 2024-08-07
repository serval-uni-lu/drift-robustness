import configutils
import h5py
import numpy as np
from mlc.load_do_save import save_json

from drift_study.reports.result_processor import load_confs
from drift_study.utils.io_utils import check_parent_path


def load_model_used(path):
    with h5py.File(path, "r") as f:
        return f["model_used"][()]


def run(
    input_dir,
) -> None:
    config = configutils.get_config()
    print(config)

    input_dir = input_dir

    conf_results = load_confs(f"./data/complete_run/{input_dir}")

    for conf in conf_results:
        conf_name = conf["config"]["runs"][0]["name"]
        val_test_idx = conf["config"]["evaluation_params"]["val_test_idx"]
        model_used = load_model_used(
            f"./data/simulator/{input_dir}/{conf_name}.hdf5"
        )
        last_idx = len(model_used)
        window_size = conf["config"]["window_size"]
        metric_idxs = [
            (window_size, last_idx),
            (window_size, val_test_idx),
            (val_test_idx, last_idx),
        ]
        n_trains = [
            len(np.unique(model_used[e[0] : e[1]])) for e in metric_idxs
        ]
        assert n_trains[0] >= n_trains[1]
        assert n_trains[0] >= n_trains[2]
        assert n_trains[0] == conf["n_train"]
        conf["n_train"] = n_trains

        out_path = (
            f"./data/complete_run_corrected/{input_dir}/{conf_name}.json"
        )
        check_parent_path(out_path)
        save_json(conf, out_path)


def run_many():

    DATASETS = ["lcld_201317_ds_time/rf_lcld", "electricity/rf_classifier"]

    DELAYS = [
        "all_delays",
        "all_delays_half",
        "all_delays_twice",
        "no_delays",
        "label_delays",
        "retraining_delays",
    ]

    for ds in DATASETS:
        for delay in DELAYS:
            run(f"{ds}/{delay}")


if __name__ == "__main__":
    run_many()
