def detect_degradation_mode(shap_features):
    """
    Detect dominant degradation mechanism
    based on top SHAP feature.
    """

    if not shap_features:
        return "Unknown"

    dominant = shap_features[0]["feature"]

    mapping = {
        "voltage_variance_mean": "Kinetic Instability (Voltage Fluctuation Driven)",
        "entropy_slope": "Thermodynamic Inefficiency",
        "irreversible_energy_ratio": "SEI Growth / Side Reactions",
        "capacity_acceleration": "Structural Degradation",
        "R_instability": "Interfacial Resistance Growth",
        "CE_instability": "Coulombic Instability"
    }

    return mapping.get(dominant, "General Degradation")
