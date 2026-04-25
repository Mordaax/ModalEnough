import numpy as np


def validate_costs(costs, budget):
    costs = np.asarray(costs, dtype=np.float32)
    if costs.shape != (3,) or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('Three finite strictly positive costs required')
    if not np.isfinite(budget) or budget < 0:
        raise ValueError('Budget must be finite and nonnegative')
    return costs
