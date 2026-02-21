def compute_confidence_score(total_cycles, instability_factor):
    """
    Confidence of prediction (0–100).
    """

    cycle_factor = min(1.0, total_cycles / 200)

    instability_penalty = max(0, instability_factor - 1)

    instability_factor_score = max(0, 1 - instability_penalty * 2)

    confidence = 0.7 * cycle_factor + 0.3 * instability_factor_score

    return round(confidence * 100, 2)
