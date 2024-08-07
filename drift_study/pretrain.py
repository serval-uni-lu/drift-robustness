import logging
import os

import configutils
import h5py
import numpy as np
from mlc.datasets.dataset_factory import get_dataset

from drift_study.utils.helpers import get_f_new_model
from drift_study.utils.io_utils import load_do_save_model

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run(
    config,
) -> None:

    dataset = get_dataset(config.get("dataset"))
    x, y, t = dataset.get_x_y_t()

    metadata = dataset.get_metadata(only_x=True)
    f_new_model = get_f_new_model(config, config, metadata)

    window_size = config.get("window_size")
    batch_size = config.get("batch_size")
    n_model = int(np.floor((len(x) - window_size) / batch_size))
    model_name = f_new_model().name

    model_root_dir = config.get(
        "models_dir", os.environ.get("MODELS_DIR", "./models")
    )
    y_preds_path = (
        f"{model_root_dir}/{dataset.name}/" f"{model_name}_preds.hdf5"
    )
    chunks = 1

    n_predict = 1
    n_class = config["evaluation_params"]["n_score"]
    y_preds_shape = (n_model, len(x), n_class)
    y_preds_dropout_shape = (n_model, len(x), n_predict, n_class)

    if not os.path.exists(y_preds_path):
        with h5py.File(y_preds_path, "w") as f:
            f.create_dataset(
                "y_preds",
                shape=y_preds_shape,
                dtype=np.float_,
                chunks=(chunks, *y_preds_shape[1:]),
                fillvalue=np.nan,
                # compression="gzip",
            )
            f.create_dataset(
                "y_preds_dropout",
                shape=y_preds_dropout_shape,
                dtype=np.float_,
                chunks=(chunks, *y_preds_dropout_shape[1:]),
                fillvalue=np.nan,
                # compression="gzip",
            )

    for model_idx in range(n_model)[:1]:

        start_idx = model_idx * batch_size
        end_idx = start_idx + window_size

        model_path = (
            f"{model_root_dir}/{dataset.name}/"
            f"{model_name}_{start_idx}_{end_idx}.joblib"
        )
        model = f_new_model()
        print(model[1].get_model_size())

        load_do_save_model(
            model,
            model_path,
            x.iloc[start_idx:end_idx],
            y[start_idx:end_idx],
        )
        # quantized_model = torch.quantization.quantize_dynamic(
        #     model[1].model, {torch.nn.Linear}, dtype=torch.qint8
        # )
        # model[1].model = quantized_model
        # model[1].device = "cpu"
        # model[1].to_device()

        with h5py.File(y_preds_path, "r+") as f:
            y_pred = model.predict_proba(x.iloc[:5000])
            f["y_preds"][model_idx, :5000] = y_pred
            model[1].train()
            for dropout_idx in range(n_predict):
                y_pred = model.predict_proba(x.iloc[:5000])
                f["y_preds_dropout"][model_idx, :5000, dropout_idx] = y_pred
                logger.info(f"Prediction {dropout_idx} done.")
            model[1].eval()
        logger.info(f"Model {model_idx} done.")


if __name__ == "__main__":
    config_all = configutils.get_config()
    run(config_all)
