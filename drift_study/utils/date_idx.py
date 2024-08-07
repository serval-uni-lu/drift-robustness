from typing import Dict

import numpy as np
import pandas as pd


def find_index_in_past(
    current_index: int, series: pd.Series, delta: Dict[str, int], past: bool
) -> int:
    current_date = series[current_index]
    if past:
        past_date = current_date - pd.DateOffset(**delta)
        return np.argwhere(series < past_date)[0][-1] + 1
    else:
        future_date = current_date + pd.DateOffset(**delta)
        return np.argwhere(series > future_date)[0][0] - 1
