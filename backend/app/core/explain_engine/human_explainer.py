def generate_human_explanation(
    fade_rate,
    resistance,
    risk_level,
    lifetime_metrics,
    shap_features
):
    """
    Converts ML + SHAP outputs into human-readable engineering insight.
    """

    explanation_lines = []

    # -------------------------------
    # 1️⃣ Degradation Severity
    # -------------------------------
    if fade_rate > -0.002:
        explanation_lines.append(
            "The battery exhibits slow degradation with stable long-term behavior."
        )
    elif fade_rate > -0.005:
        explanation_lines.append(
            "The battery shows moderate degradation and may experience accelerated aging after mid-life cycles."
        )
    else:
        explanation_lines.append(
            "The battery demonstrates aggressive degradation and early capacity decline."
        )

    # -------------------------------
    # 2️⃣ Resistance Insight
    # -------------------------------
    if resistance < 150:
        explanation_lines.append(
            "Internal resistance remains low, indicating good electrochemical kinetics."
        )
    elif resistance < 300:
        explanation_lines.append(
            "Moderate internal resistance suggests developing interfacial impedance."
        )
    else:
        explanation_lines.append(
            "High internal resistance indicates significant kinetic limitations or structural degradation."
        )

    # -------------------------------
    # 3️⃣ Lifetime Commentary
    # -------------------------------
    eol = lifetime_metrics.get("estimated_end_of_life_cycles")

    if eol:
        explanation_lines.append(
            f"Projected end-of-life is approximately {eol} cycles at 80% capacity retention."
        )

    # -------------------------------
    # 4️⃣ SHAP Root Cause Analysis
    # -------------------------------
    dominant = shap_features[0]["feature"] if shap_features else None

    feature_mapping = {
        "voltage_variance_mean": "voltage instability",
        "entropy_slope": "thermodynamic inefficiency",
        "irreversible_energy_ratio": "energy loss mechanisms",
        "capacity_acceleration": "nonlinear degradation acceleration",
        "R_instability": "resistance fluctuation",
        "CE_instability": "coulombic inefficiency"
    }

    if dominant in feature_mapping:
        explanation_lines.append(
            f"The dominant degradation driver appears to be {feature_mapping[dominant]}."
        )

    # -------------------------------
    # 5️⃣ Risk Summary
    # -------------------------------
    explanation_lines.append(
        f"Overall system risk level is classified as {risk_level}."
    )

    return " ".join(explanation_lines)
