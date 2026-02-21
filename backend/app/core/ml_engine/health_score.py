import math


def normalize(value, min_val, max_val):
    """
    Normalizes value between 0 and 1.
    """
    if value is None:
        return 0.5

    value = max(min_val, min(max_val, value))
    return (value - min_val) / (max_val - min_val)


def compute_health_score(
    fade_rate,
    resistance,
    instability_factor,
    eol_cycles
):
    """
    Advanced battery health scoring system (0–100).

    Factors:
    - Degradation rate
    - Internal resistance
    - Lifetime projection
    - Instability penalty
    """

    # ----------------------------
    # 1️⃣ Fade Component (most important)
    # Ideal fade ~ 0
    # Aggressive fade < -0.01
    # ----------------------------
    fade_severity = abs(fade_rate)

    fade_score = 1 - normalize(fade_severity, 0.0, 0.01)

    # ----------------------------
    # 2️⃣ Resistance Component
    # Ideal < 100
    # Bad > 500
    # ----------------------------
    resistance_score = 1 - normalize(resistance, 100, 500)

    # ----------------------------
    # 3️⃣ Lifetime Component
    # Ideal > 1200 cycles
    # Bad < 200 cycles
    # ----------------------------
    if eol_cycles is None:
        lifetime_score = 0.5
    else:
        lifetime_score = normalize(eol_cycles, 200, 1200)

    # ----------------------------
    # 4️⃣ Instability Penalty
    # instability_factor = 1 is stable
    # >1 means unstable
    # ----------------------------
    instability_penalty = max(0, instability_factor - 1)

    instability_score = max(0, 1 - instability_penalty * 2)

    # ----------------------------
    # 5️⃣ Weighted Aggregation
    # ----------------------------
    total_score = (
        0.40 * fade_score +
        0.25 * resistance_score +
        0.20 * lifetime_score +
        0.15 * instability_score
    )

    return round(total_score * 100, 2)
