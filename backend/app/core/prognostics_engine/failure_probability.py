import math


def compute_failure_probability(current_cycles, eol_cycles):
    """
    Logistic failure probability model.
    """

    if eol_cycles is None or eol_cycles == 0:
        return 50.0

    ratio = current_cycles / eol_cycles

    # Logistic curve centered at 1.0
    prob = 1 / (1 + math.exp(-8 * (ratio - 1)))

    return round(prob * 100, 2)
