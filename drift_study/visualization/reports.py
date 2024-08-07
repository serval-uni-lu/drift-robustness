import os


def command_figure(use_case: str, max_n_train: int, delay: str) -> str:
    return (
        f"python -m drift_study.plot_ml_ntrain "
        f"-p input_dir=./data/complete_run/{use_case}/{delay} "
        f"-p output_file=./reports_max_train/{use_case}/{delay}.pdf "
        f"-p max_n_train={max_n_train}"
    )


def command_interactive_figure(
    use_case: str, max_n_train: int, delay: str
) -> str:
    return (
        f"python -m drift_study.plot_ml_ntrain "
        f"-p input_dir=./data/complete_run/{use_case}/{delay} "
        f"-p output_file=./reports_max_train/{use_case}/{delay}.html "
        f"-p plot_engine=plotly "
        f"-p max_n_train={max_n_train}"
    )


def command_table(use_case: str, max_n_train: int, delay: str) -> str:
    return (
        f"python -m drift_study.table_ml_ntrain "
        f"-p input_dir=./data/complete_run/{use_case}/{delay} "
        f"-p output_file=./reports_max_train/{use_case}/{delay}.csv "
        f"-p max_n_train={max_n_train}"
    )


USE_CASES = [
    ("lcld_201317_ds_time/rf_lcld", 165),
    ("electricity/rf_classifier", 31),
]
DELAYS = ["no_delays", "all_delays", "all_delays_half", "all_delays_twice"]
COMMANDS = [command_table, command_figure, command_interactive_figure]


def run():
    for command in COMMANDS:
        for use_case in USE_CASES:
            for delay in DELAYS:
                os.system(command(use_case[0], use_case[1], delay))


if __name__ == "__main__":
    run()
