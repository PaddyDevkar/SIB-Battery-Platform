from app.config import (
    IDEAL_INTERNAL_RESISTANCE,
    EXPECTED_MAX_RESISTANCE
)


def compute_energy_soh(current_cycle, projected_curve):
    """
    SOH based on projected capacity at current cycle.
    """
    capacities = projected_curve["projected_capacity"]

    if current_cycle >= len(capacities):
        current_capacity = capacities[-1]
    else:
        current_capacity = capacities[current_cycle]

    return round(max(0, current_capacity) * 100, 2)


def compute_power_soh(resistance):
    """
    SOH based on internal resistance.
    """
    if resistance <= 0:
        return 0

    ratio = IDEAL_INTERNAL_RESISTANCE / resistance

    return round(min(max(ratio * 100, 0), 100), 2)


def compute_overall_soh(energy_soh, power_soh):
    """
    Combined SOH metric.
    """
    return round((0.6 * energy_soh + 0.4 * power_soh), 2)


def compute_remaining_useful_life(current_cycles, eol_cycles):
    """
    RUL in cycles.
    """
    if eol_cycles is None:
        return None

    return max(0, eol_cycles - current_cycles)


def classify_degradation_phase(current_cycles, eol_cycles):
    """
    Lifecycle stage classification.
    """

    if eol_cycles is None or eol_cycles == 0:
        return "Unknown"

    ratio = current_cycles / eol_cycles

    if ratio < 0.3:
        return "Early Life"
    elif ratio < 0.7:
        return "Stable Mid-Life"
    elif ratio < 1.0:
        return "Knee Region / Accelerated Aging"
    else:
        return "End-of-Life"
