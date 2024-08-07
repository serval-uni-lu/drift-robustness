from pathlib import Path
from typing import Any, Dict, List

import configutils
import numpy as np
import pandas as pd

from drift_study.reports.graphics import scatterplot
from drift_study.reports.naming import beautify_dataframe
from drift_study.reports.result_processor import filter_conf, load_confs
from drift_study.utils.pareto import calc_pareto_rank


def filter_extreme(
    confs: List[Dict[str, Any]], extreme_value: int
) -> List[Dict[str, Any]]:
    out = [conf for conf in confs if conf["n_train"] <= extreme_value]
    out = [conf for conf in out if (conf["n_train"] > 1) or (conf["detector_type"] == "baseline")]
    return out


def run() -> None:
    config = configutils.get_config()
    print(config)

    input_dir = config["input_dir"]
    output_file = config.get("output_file", None)
    plot_engine = config.get("plot_engine", "sns")

    conf_results = load_confs(input_dir)
    conf_results = [filter_conf(e) for e in conf_results]

    max_n_train = config.get("max_n_train", -1)
    if max_n_train > 0:
            conf_results = filter_extreme(conf_results, max_n_train)
    df = pd.DataFrame(conf_results)

    # Add rank
    pareto_rank = calc_pareto_rank(
        np.array([df["n_train"], df["ml_metric"]]).T, np.array([1, -1])
    )
    df["pareto_rank"] = pareto_rank

    df = beautify_dataframe(df.copy())

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if plot_engine == "sns":

        scatterplot(
            df,
            output_file,
            x="n_train",
            y="ml_metric",
            y_label=conf_results[0]["metric_name"].upper(),
            hue="Type",
            x_label="\\# Train",
            fig_size=(6, 4),
            legend_pos="best",
            markers=["o", "s", "^", "x"],
        )
    elif plot_engine == "plotly":
        import plotly.express as px
        
        max_pareto_rank = config.get("max_pareto_rank", -1)
        if max_pareto_rank > 0:
            df = df[(df["pareto_rank"] <= max_pareto_rank) | (df["detector_type"] == "baseline")]
        
        df = df
        fig = px.scatter(
            df,
            x="n_train",
            y="ml_metric",
            # color="pareto_rank",
            symbol="detector_type",
            hover_data=["detector_name", "pareto_rank"],
        )
        fig.write_html(output_file)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    run()
