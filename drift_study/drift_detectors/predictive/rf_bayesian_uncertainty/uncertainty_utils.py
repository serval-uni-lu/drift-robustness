from typing import Dict, List

import numpy as np
import numpy.typing as npt


def calculate_entropy_uncertainties(
    labels: List[int],
    list_end_leafs: npt.NDArray[np.int_],
    leafs_split: List[Dict[int, List[int]]],
):

    max_leaf_n = max([max([*e.keys()]) for e in leafs_split]) + 1
    np_leafs_split = np.full(
        ((len(leafs_split)), max_leaf_n, len(labels)), np.nan
    )
    for i, e in enumerate(leafs_split):
        np_leafs_split[i, [*e.keys()]] = np.array([*e.values()])

    n_labels = len(labels)

    n_y = np_leafs_split[np.arange(len(np_leafs_split)), list_end_leafs]
    n_y_s = np.sum(n_y, axis=2).reshape(-1, len(np_leafs_split), 1)

    class_conditional_probabilities = (n_y + 1) / (n_y_s + n_labels)
    p = class_conditional_probabilities
    p_log_p = p * np.log2(p)
    # tot_p = np.sum(p, axis=1)
    # tot_p_log_p = np.sum(p_log_p, axis=1)

    mean_tot_p = np.mean(p, axis=1)
    mean_tot_p_log_p = np.mean(p_log_p, axis=1) / len(leafs_split)

    log_mean_tot_p = np.log2(mean_tot_p)
    tot_u_new = np.sum(-mean_tot_p * log_mean_tot_p, axis=1)
    al_u_new = np.sum(-mean_tot_p_log_p, axis=1)

    return tot_u_new, tot_u_new - al_u_new, al_u_new
