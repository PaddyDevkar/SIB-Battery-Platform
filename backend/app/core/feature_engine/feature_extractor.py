import numpy as np


def extract_features(cycle_df, raw_df):
    """
    Extract features from EARLY cycles (<= 30).
    Must match training logic exactly.
    """

    # Use early cycles only
    cycle_df = cycle_df[cycle_df["Cycle_Index"] <= 30]

    if cycle_df is None or len(cycle_df) < 10:
        return None

    cycles = cycle_df["Cycle_Index"].values
    cap = cycle_df["discharge_capacity"].values
    ce = cycle_df["CE"].values
    voltage = cycle_df["voltage_mean"].values
    v_std = cycle_df["voltage_std"].values
    energy_loss = cycle_df["energy_loss"].values

    features = {}

    # -------------------------------
    # 1️⃣ Capacity Domain
    # -------------------------------
    features["initial_capacity"] = cap[0]
    features["final_capacity"] = cap[-1]
    features["capacity_slope"] = np.polyfit(cycles, cap, 1)[0]
    features["capacity_curvature"] = np.polyfit(cycles, cap, 2)[0]
    features["capacity_acceleration"] = np.polyfit(cycles, cap, 3)[0]
    features["capacity_noise"] = np.std(cap)
    features["capacity_range"] = np.max(cap) - np.min(cap)
    features["capacity_drop_ratio"] = (
        (cap[0] - cap[-1]) / cap[0] if cap[0] != 0 else np.nan
    )

    mid = len(cap) // 2
    features["early_capacity_slope"] = np.polyfit(cycles[:mid], cap[:mid], 1)[0]
    features["late_capacity_slope"] = np.polyfit(cycles[mid:], cap[mid:], 1)[0]

    # -------------------------------
    # 2️⃣ Coulombic Efficiency
    # -------------------------------
    valid_ce = ce[~np.isnan(ce)]

    if len(valid_ce) > 5:
        features["CE_mean"] = np.mean(valid_ce)
        features["CE_std"] = np.std(valid_ce)
        features["CE_slope"] = np.polyfit(cycles[:len(valid_ce)], valid_ce, 1)[0]
        features["CE_instability"] = np.std(np.diff(valid_ce))
        features["CE_min"] = np.min(valid_ce)
        features["CE_max"] = np.max(valid_ce)
    else:
        features["CE_mean"] = np.nan
        features["CE_std"] = np.nan
        features["CE_slope"] = np.nan
        features["CE_instability"] = np.nan
        features["CE_min"] = np.nan
        features["CE_max"] = np.nan

    # -------------------------------
    # 3️⃣ Voltage Domain
    # -------------------------------
    features["voltage_slope"] = np.polyfit(cycles, voltage, 1)[0]
    features["voltage_curvature"] = np.polyfit(cycles, voltage, 2)[0]
    features["voltage_variance_mean"] = np.mean(v_std)
    features["voltage_instability"] = np.std(np.diff(voltage))
    features["voltage_range"] = np.max(voltage) - np.min(voltage)

    # -------------------------------
    # 4️⃣ Thermodynamic Domain
    # -------------------------------
    T = 298
    entropy = (energy_loss * 3600) / T

    features["entropy_mean"] = np.mean(entropy)
    features["entropy_slope"] = np.polyfit(cycles, entropy, 1)[0]
    features["entropy_curvature"] = np.polyfit(cycles, entropy, 2)[0]
    features["entropy_std"] = np.std(entropy)

    total_charge_energy = np.sum(cycle_df["charge_energy"])
    features["irreversible_energy_ratio"] = (
        np.sum(energy_loss) / total_charge_energy
        if total_charge_energy != 0 else np.nan
    )

    # -------------------------------
    # 5️⃣ Resistance Domain
    # -------------------------------
    raw_df = raw_df.copy()
    raw_df["dV"] = raw_df["Voltage(V)"].diff()
    raw_df["dI"] = raw_df["Current(A)"].diff()

    valid = raw_df["dI"] != 0
    R = (raw_df["dV"][valid] / raw_df["dI"][valid]).dropna()

    if len(R) > 10:
        features["R_mean"] = R.mean()
        features["R_std"] = R.std()
        features["R_slope"] = np.polyfit(np.arange(len(R)), R, 1)[0]
        features["R_instability"] = np.std(np.diff(R))
    else:
        features["R_mean"] = np.nan
        features["R_std"] = np.nan
        features["R_slope"] = np.nan
        features["R_instability"] = np.nan

    # -------------------------------
    # 6️⃣ Instability Metrics
    # -------------------------------
    features["capacity_volatility"] = np.std(np.diff(cap))
    features["voltage_volatility"] = np.std(np.diff(voltage))
    features["combined_instability_index"] = (
        features["capacity_volatility"] *
        features["voltage_volatility"]
    )

    return features
