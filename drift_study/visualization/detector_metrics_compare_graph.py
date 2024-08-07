import logging
import os

import configutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from configutils.utils import merge_parameters

from drift_study import detector_metrics_compare

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run():
    config = configutils.get_config()
    if config.get("metric_path") is not None:
        df = pd.read_csv(config.get("metric_path"))
    else:
        df = detector_metrics_compare.run()

    print(df)
    ax = sns.scatterplot(
        df, x="n_train", y="metric", hue="pareto_front", style="type"
    )
    metric_name = config.get("evaluation_params").get("metric").get("name")
    ax.set(ylabel=f"Loss in {metric_name}")
    dataset_name = config.get("dataset").get("name")

    for i in range(len(config.get("runs"))):
        config.get("runs")[i] = merge_parameters(
            config.get("common_runs_params").copy(),
            config.get("runs")[i].copy(),
        )

    model_name = config.get("runs")[0].get("model").get("name")
    offset = 0.02 * (df["metric"].max() - df["metric"].min())

    def plotlabel(xvar, yvar, label, alternate=0):
        ax.text(xvar + 0.2, yvar + alternate * offset, label)

    df_show_text = df[df["pareto_front"] == 1]
    df_show_text = df_show_text.sort_values(by="n_train")
    df_show_text["alternate"] = np.tile(np.array([-2, 1]), len(df_show_text))[
        : len(df_show_text)
    ]

    df_show_text_gr = df_show_text.groupby(
        ["n_train", "metric"], as_index=False
    ).agg(
        {
            "method_name": lambda x: " / ".join(x.tolist()),
            "pareto_front": "min",
            "alternate": lambda x: x.tolist()[-1],
        }
    )

    df_show_text_gr.apply(
        lambda x: plotlabel(
            x["n_train"],
            x["metric"],
            x["method_name"],
            x["alternate"],
        ),
        axis=1,
    )
    plt.show()
    plt.savefig(
        f"./reports/{dataset_name}/{model_name}_{metric_name}_compare.pdf"
    )


if __name__ == "__main__":
    run()
