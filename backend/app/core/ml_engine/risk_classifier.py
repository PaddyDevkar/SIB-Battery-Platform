def classify_risk(predicted_fade_rate: float):
    """
    Classify degradation risk based on normalized fade rate.
    More negative fade rate = faster degradation.
    """

    # Example thresholds (can be tuned later)
    if predicted_fade_rate > -0.001:
        return "Low"

    elif predicted_fade_rate > -0.005:
        return "Medium"

    else:
        return "High"
