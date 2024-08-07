from typing import Optional

import numpy as np
import numpy.typing as npt


def is_pareto_efficient_simple(
    costs: npt.NDArray[np.float_],
) -> npt.NDArray[np.bool_]:
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array,
        indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def calc_pareto_rank(
    costs: npt.NDArray[np.float_],
    costs_direction: Optional[npt.NDArray[np.float_]],
) -> npt.NDArray[np.int_]:

    if costs_direction is None:
        costs_direction = np.ones(costs.shape[1])

    costs = costs * costs_direction
    costs, inverse, count = np.unique(
        costs, return_inverse=True, return_counts=True, axis=0
    )
    n = len(costs)

    to_front_idx = np.arange(n)
    rank = np.full(n, 1e16, dtype=int)
    i = 1
    while len(to_front_idx) > 0:
        current_cost = costs[to_front_idx]
        efficient = is_pareto_efficient_simple(current_cost)
        rank[to_front_idx[efficient]] = i
        to_front_idx = to_front_idx[~efficient]
        i += 1

    rank = rank[inverse]
    return rank
