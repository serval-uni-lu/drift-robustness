from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def sample_date(
    x: pd.DataFrame,
    y: npt.NDArray[Union[np.int_, np.float_]],
    t: Union[pd.Series, npt.NDArray[np.int_]],
    minority_share: Optional[float],
) -> Tuple[
    pd.DataFrame,
    npt.NDArray[Union[np.int_, np.float_]],
    Union[pd.Series, npt.NDArray[np.int_]],
]:
    if minority_share is None:
        return x, y, t

    sampling_strategy = minority_share / (1 - minority_share)
    idx = np.arange(len(x))
    sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=42
    )
    idx_new, _ = sampler.fit_resample(idx.reshape(-1, 1), y)
    idx_new = np.sort(idx_new.flatten())

    return (
        x.iloc[idx_new].reset_index(drop=True),
        y[idx_new],
        t[idx_new].reset_index(drop=True),
    )
