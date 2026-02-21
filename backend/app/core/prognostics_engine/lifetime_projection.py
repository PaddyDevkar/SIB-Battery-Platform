import numpy as np


# 🔥 Projection calibration factor
# Converts ML fade index → realistic physical degradation
CALIBRATION_FACTOR = 0.1


def project_capacity_curve(
    fade_rate,
    initial_capacity=1.0,
    max_cycles=2000,
    instability_factor=1.0
):
    """
    Calibrated two-stage degradation model.

    ML fade_rate is treated as degradation index.
    Converted to realistic physical fade using calibration factor.
    """

    # ----------------------------------------------------
    # 1️⃣ Calibrate fade rate
    # ----------------------------------------------------
    effective_fade = fade_rate * CALIBRATION_FACTOR

    # Safety clamp (prevents absurd collapse)
    effective_fade = max(min(effective_fade, 0), -0.002)

    cycles = np.arange(0, max_cycles)

    # ----------------------------------------------------
    # 2️⃣ Adaptive knee location
    # ----------------------------------------------------
    if effective_fade == 0:
        knee_point = max_cycles
    else:
        base_knee = int(1 / (abs(effective_fade) * 4))
        knee_point = max(100, int(base_knee / instability_factor))

    capacities = []

    for c in cycles:

        # Stage 1 — Stable degradation
        if c <= knee_point:
            cap = initial_capacity * (1 + effective_fade * c)

        # Stage 2 — Accelerated degradation
        else:
            stage1_cap = initial_capacity * (1 + effective_fade * knee_point)

            accelerated_fade = effective_fade * 2  # softer acceleration

            cap = stage1_cap + accelerated_fade * (c - knee_point)

        capacities.append(max(cap, 0))  # prevent negative capacity

    return {
        "cycles": cycles.tolist(),
        "projected_capacity": capacities,
        "predicted_knee_cycle": knee_point
    }

def estimate_eol_cycles(
    fade_rate,
    threshold=0.8,
    initial_capacity=1.0,
    max_limit=5000,
    instability_factor=1.0
):
    """
    Estimate cycles to reach capacity threshold
    using calibrated degradation model.
    """

    curve = project_capacity_curve(
        fade_rate,
        initial_capacity=initial_capacity,
        max_cycles=max_limit,
        instability_factor=instability_factor
    )

    capacities = np.array(curve["projected_capacity"])

    idx = np.where(capacities <= threshold)[0]

    if len(idx) == 0:
        return None

    return int(idx[0])


def full_lifetime_analysis(
    fade_rate,
    instability_factor=1.0
):
    """
    Structured lifetime prediction.
    """

    cycles_80 = estimate_eol_cycles(
        fade_rate,
        threshold=0.8,
        instability_factor=instability_factor
    )

    cycles_70 = estimate_eol_cycles(
        fade_rate,
        threshold=0.7,
        instability_factor=instability_factor
    )

    cycles_60 = estimate_eol_cycles(
        fade_rate,
        threshold=0.6,
        instability_factor=instability_factor
    )

    return {
        "cycles_to_80_percent": cycles_80,
        "cycles_to_70_percent": cycles_70,
        "cycles_to_60_percent": cycles_60,
        "estimated_end_of_life_cycles": cycles_80
    }
