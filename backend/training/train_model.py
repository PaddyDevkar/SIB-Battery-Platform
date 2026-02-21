# Import
import zipfile
import os
import pandas as pd
import re
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

zip_path = "/Users/pranavdevkar/Downloads/Na-upscaling.zip"
extract_path = "/Users/pranavdevkar/Downloads/Na_upscaling_extracted"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("ZIP extracted successfully.")

for root, dirs, files in os.walk(extract_path):
    print("Folder:", root)
    break


def scan_files(root_dir):
    records = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:

            if not (file.lower().endswith(".csv") or file.lower().endswith(".hdf5")):
                continue

            full_path = os.path.join(root, file)

            records.append({
                "file_name": file,
                "file_path": full_path,
                "extension": file.split('.')[-1].lower(),
                "folder": os.path.basename(root),
                "file_size_MB": round(os.path.getsize(full_path) / (1024 ** 2), 2)
            })

    return pd.DataFrame(records)


registry_df = scan_files(extract_path)

print("Total data files found:", len(registry_df))
print(registry_df.head())


def extract_cell_id(filename):
    filename = filename.lower()

    # Format 1: cell138
    match1 = re.search(r'cell(\d+)', filename)
    if match1:
        return int(match1.group(1))

    # Format 2: _67_channel
    match2 = re.search(r'_(\d+)_channel', filename)
    if match2:
        return int(match2.group(1))

    return None


registry_df["cell_id"] = registry_df["file_name"].apply(extract_cell_id)

print("Unique cells detected:", registry_df["cell_id"].nunique())

registry_df = registry_df.sort_values(by="file_path").reset_index(drop=True)
registry_df["File_ID"] = range(1, len(registry_df) + 1)
unique_cells = sorted(registry_df["cell_id"].dropna().unique())

cell_to_system = {
    cell: i+1 for i, cell in enumerate(unique_cells)
}

registry_df["System_ID"] = registry_df["cell_id"].map(cell_to_system)

print("Total Systems:", registry_df["System_ID"].nunique())

registry_df = registry_df[
    ["File_ID",
     "System_ID",
     "cell_id",
     "file_name",
     "extension",
     "folder",
     "file_size_MB",
     "file_path"]
]
registry_df.to_csv("Layer0_FileRegistry.csv", index=False)
print("Layer 0 registry saved.")

pd.set_option('display.float_format', '{:.8e}'.format)


def load_file(file_path):
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)

        df.rename(columns={
            "Test_Time(s)": "Test_Time",
            "Charge_Energy(Wh)": "Charge_Energy",
            "Discharge_Energy(Wh)": "Discharge_Energy"
        }, inplace=True)

        return df


    elif file_path.lower().endswith(".hdf5"):
        with h5py.File(file_path, 'r') as f:
            raw = f["raw"]

            current = raw["Current(A)"][:]
            voltage = raw["Voltage(V)"][:]
            time = raw["Testtime(s)"][:]
            cycle = raw["Cycle_Index"][:]

            min_len = min(len(current), len(voltage), len(time), len(cycle))

            df = pd.DataFrame({
                "Current(A)": current[:min_len],
                "Voltage(V)": voltage[:min_len],
                "Test_Time": time[:min_len],
                "Cycle_Index": cycle[:min_len]
            })

        return df

    else:
        return None


def clean_signal(df):
    required = ["Voltage(V)", "Current(A)", "Test_Time"]

    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=required)
    df = df[df["Voltage(V)"] > 0]
    df = df.sort_values("Test_Time").reset_index(drop=True)

    return df


def regenerate_cycles(df):
    if "Cycle_Index" not in df.columns:
        df["Cycle_Index"] = 0

    # If cycle index is unrealistic (like 100000+)
    if df["Cycle_Index"].max() > 10000:
        df = df.copy()

        sign = np.sign(df["Current(A)"])
        transitions = (sign.diff() != 0).astype(int)
        df["Cycle_Index"] = transitions.cumsum()

    return df


def compute_capacity_energy(df):
    df = df.copy()

    df["dt"] = df["Test_Time"].diff().fillna(0)

    # Capacity (Ah)
    df["dAh"] = df["Current(A)"] * df["dt"] / 3600.0

    # Energy (Wh)
    df["dWh"] = df["Voltage(V)"] * df["Current(A)"] * df["dt"] / 3600.0

    df["Charge_Capacity(Ah)"] = 0.0
    df["Discharge_Capacity(Ah)"] = 0.0
    df["Charge_Energy"] = 0.0
    df["Discharge_Energy"] = 0.0

    for cycle in df["Cycle_Index"].unique():
        mask = df["Cycle_Index"] == cycle
        sub = df.loc[mask]

        charge_cap = sub["dAh"].where(sub["Current(A)"] > 0, 0).cumsum()
        discharge_cap = (-sub["dAh"].where(sub["Current(A)"] < 0, 0)).cumsum()

        charge_energy = sub["dWh"].where(sub["Current(A)"] > 0, 0).cumsum()
        discharge_energy = (-sub["dWh"].where(sub["Current(A)"] < 0, 0)).cumsum()

        df.loc[mask, "Charge_Capacity(Ah)"] = charge_cap.values
        df.loc[mask, "Discharge_Capacity(Ah)"] = discharge_cap.values
        df.loc[mask, "Charge_Energy"] = charge_energy.values
        df.loc[mask, "Discharge_Energy"] = discharge_energy.values

    return df


def generate_cycle_summary(df):
    """
    Convert time-series dataframe into cycle-level summary.
    Requires:
        - Charge_Capacity(Ah)
        - Discharge_Capacity(Ah)
        - Charge_Energy
        - Discharge_Energy
        - Voltage(V)
        - Current(A)
        - Cycle_Index
    """

    # Ensure enough cycles
    if "Cycle_Index" not in df.columns:
        return None

    if df["Cycle_Index"].nunique() < 5:
        return None

    # ----------------------------
    # Per-cycle aggregation
    # ----------------------------
    cycle_df = df.groupby("Cycle_Index").agg({
        "Charge_Capacity(Ah)": "max",
        "Discharge_Capacity(Ah)": "max",
        "Charge_Energy": "sum",
        "Discharge_Energy": "sum",
        "Voltage(V)": "mean",
        "Current(A)": "mean"
    }).reset_index()

    # Rename for clarity
    cycle_df.rename(columns={
        "Charge_Capacity(Ah)": "charge_capacity",
        "Discharge_Capacity(Ah)": "discharge_capacity",
        "Charge_Energy": "charge_energy",
        "Discharge_Energy": "discharge_energy",
        "Voltage(V)": "voltage_mean",
        "Current(A)": "current_mean"
    }, inplace=True)

    # ----------------------------
    # Compute voltage std manually
    # (avoids 77% NaN problem)
    # ----------------------------
    voltage_std = (
        df.groupby("Cycle_Index")["Voltage(V)"]
        .std()
        .fillna(0)
        .values
    )

    cycle_df["voltage_std"] = voltage_std

    # ----------------------------
    # Coulombic Efficiency
    # ----------------------------
    cycle_df["CE"] = np.where(
        cycle_df["charge_capacity"] > 0,
        cycle_df["discharge_capacity"] /
        cycle_df["charge_capacity"],
        np.nan
    )

    # ----------------------------
    # Energy Loss per Cycle
    # ----------------------------
    cycle_df["energy_loss"] = (
            cycle_df["charge_energy"] -
            cycle_df["discharge_energy"]
    )

    # ----------------------------
    # Voltage Hysteresis Proxy
    # (difference between charge and discharge voltage)
    # ----------------------------
    charge_voltage = (
        df[df["Current(A)"] > 0]
        .groupby("Cycle_Index")["Voltage(V)"]
        .mean()
    )

    discharge_voltage = (
        df[df["Current(A)"] < 0]
        .groupby("Cycle_Index")["Voltage(V)"]
        .mean()
    )

    hysteresis = (
            charge_voltage - discharge_voltage
    ).reindex(cycle_df["Cycle_Index"]).fillna(0)

    cycle_df["voltage_hysteresis"] = hysteresis.values

    # ----------------------------
    # Capacity retention
    # ----------------------------
    if cycle_df["discharge_capacity"].iloc[0] > 0:
        cycle_df["capacity_retention"] = (
                cycle_df["discharge_capacity"] /
                cycle_df["discharge_capacity"].iloc[0]
        )
    else:
        cycle_df["capacity_retention"] = np.nan

    return cycle_df


def extract_features(cycle_df, raw_df):
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
    features["capacity_drop_ratio"] = (cap[0] - cap[-1]) / cap[0] if cap[0] != 0 else np.nan

    # Early vs late slope
    mid = len(cap) // 2
    features["early_capacity_slope"] = np.polyfit(cycles[:mid], cap[:mid], 1)[0]
    features["late_capacity_slope"] = np.polyfit(cycles[mid:], cap[mid:], 1)[0]

    # -------------------------------
    # 2️⃣ Coulombic Efficiency Domain
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

    features["irreversible_energy_ratio"] = (
        np.sum(energy_loss) / np.sum(cycle_df["charge_energy"])
        if np.sum(cycle_df["charge_energy"]) != 0 else np.nan
    )

    # -------------------------------
    # 5️⃣ Resistance Domain
    # -------------------------------
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


layer1_results = []

for _, row in tqdm(registry_df.iterrows(), total=len(registry_df)):

    file_id = row["File_ID"]
    system_id = row["System_ID"]
    file_path = row["file_path"]

    df = load_file(file_path)
    if df is None:
        continue

    df = clean_signal(df)
    if df is None:
        continue

    df = regenerate_cycles(df)
    df = compute_capacity_energy(df)

    cycle_df = generate_cycle_summary(df)
    feats = extract_features(cycle_df, df)

    if feats is None:
        continue

    feats["File_ID"] = file_id
    feats["System_ID"] = system_id

    layer1_results.append(feats)

layer1_df = pd.DataFrame(layer1_results)

print("Layer 1 Completed")
print("Total valid samples:", len(layer1_df))
layer1_df.head()

layer1_df = layer1_df.loc[:, layer1_df.isna().mean() < 0.2]
low_var = layer1_df.drop(columns=["File_ID","System_ID"]).std()
low_var_cols = low_var[low_var < 1e-10].index.tolist()

layer1_df = layer1_df.drop(columns=low_var_cols)
import numpy as np

corr = layer1_df.drop(columns=["File_ID","System_ID"]).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

layer1_df = layer1_df.drop(columns=to_drop)

print("Final shape:", layer1_df.shape)

# Separate identifiers
ids = layer1_df[["File_ID", "System_ID"]].reset_index(drop=True)

# Feature columns only
feature_cols = [col for col in layer1_df.columns
                if col not in ["File_ID", "System_ID"]]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(layer1_df[feature_cols])

scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

block1_features = [
    "initial_capacity",
    "final_capacity",
    "capacity_slope",
    "early_capacity_slope",
    "late_capacity_slope",
    "capacity_curvature",
    "capacity_acceleration",
    "capacity_volatility"
]

block2_features = [
    "CE_mean",
    "CE_std",
    "CE_instability"
]

block3_features = [
    "voltage_slope",
    "voltage_instability",
    "voltage_hysteresis",
    "voltage_range"
]

block4_features = [
    "R_mean",
    "R_slope",
    "R_instability"
]

block5_features = [
    "entropy_slope",
    "entropy_curvature",
    "irreversible_energy_ratio"
]

def safe_block_mean(scaled_df, feature_list):
    existing = [f for f in feature_list if f in scaled_df.columns]
    if len(existing) == 0:
        return None
    return scaled_df[existing].mean(axis=1)

layer2_df = pd.DataFrame()

layer2_df["Block_1"] = safe_block_mean(scaled_df, [
    "initial_capacity",
    "final_capacity",
    "capacity_slope",
    "early_capacity_slope",
    "late_capacity_slope",
    "capacity_curvature",
    "capacity_acceleration",
    "capacity_volatility"
])

layer2_df["Block_2"] = safe_block_mean(scaled_df, [
    "CE_mean",
    "CE_std",
    "CE_instability"
])

layer2_df["Block_3"] = safe_block_mean(scaled_df, [
    "voltage_slope",
    "voltage_instability",
    "voltage_hysteresis",
    "voltage_range"
])

layer2_df["Block_4"] = safe_block_mean(scaled_df, [
    "R_mean",
    "R_slope",
    "R_instability"
])

layer2_df["Block_5"] = safe_block_mean(scaled_df, [
    "entropy_slope",
    "entropy_curvature",
    "irreversible_energy_ratio"
])

layer2_df["File_ID"] = ids["File_ID"]
layer2_df["System_ID"] = ids["System_ID"]

layer2_df.head()

# ===============================
# BUILD TRAINING DATA
# ===============================

X_list = []
y_list = []

for _, row in registry_df.iterrows():

    file_path = row["file_path"]

    df = load_file(file_path)
    if df is None:
        continue

    df = clean_signal(df)
    if df is None:
        continue

    df = regenerate_cycles(df)
    df = compute_capacity_energy(df)

    cycle_df = generate_cycle_summary(df)
    if cycle_df is None:
        continue

    # ----- EARLY CYCLES -----
    early_cycle_df = cycle_df[cycle_df["Cycle_Index"] <= 30]
    if len(early_cycle_df) < 10:
        continue

    early_feats = extract_features(early_cycle_df, df)

    # ----- FULL CYCLES -----
    full_feats = extract_features(cycle_df, df)

    if early_feats is None or full_feats is None:
        continue

    if full_feats["initial_capacity"] == 0:
        continue

    normalized_fade = (
        full_feats["capacity_slope"] /
        full_feats["initial_capacity"]
    )

    R_mean = full_feats["R_mean"]

    if np.isnan(normalized_fade) or np.isnan(R_mean):
        continue

    X_list.append(early_feats)
    y_list.append({
        "normalized_fade_rate": normalized_fade,
        "R_mean": R_mean
    })

# ===============================
# CONVERT TO DATAFRAME
# ===============================

X = pd.DataFrame(X_list)
y = pd.DataFrame(y_list)

print("Final training shape:", X.shape, y.shape)

# ===============================
# INITIALIZE SCALER + MODEL
# ===============================

scaler = StandardScaler()

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=400,
        max_depth=6,
        random_state=42
    )
)

# ===============================
# SCALE (KEEP AS DATAFRAME)
# ===============================

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# ===============================
# CROSS VALIDATION (USE SCALED DATA)
# ===============================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []

for train_idx, test_idx in kf.split(X_scaled):

    model.fit(
        X_scaled.iloc[train_idx],
        y.iloc[train_idx]
    )

    preds = model.predict(
        X_scaled.iloc[test_idx]
    )

    r2 = r2_score(
        y.iloc[test_idx],
        preds,
        multioutput='raw_values'
    )

    r2_scores.append(r2)

r2_scores = np.array(r2_scores)

print("Average R2 per target:")
print("normalized_fade_rate :", r2_scores.mean(axis=0)[0])
print("R_mean               :", r2_scores.mean(axis=0)[1])

# ===============================
# FINAL TRAINING ON FULL DATA
# ===============================

model.fit(X_scaled, y)

train_preds = model.predict(X_scaled)

print("Training R2:",
      r2_score(y, train_preds, multioutput='raw_values'))

# ===============================
# FREEZE MODEL
# ===============================

feature_columns = X.columns.tolist()

import joblib
import os

os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/sib_model.pkl")
joblib.dump(scaler, "../models/sib_scaler.pkl")
joblib.dump(feature_columns, "../models/sib_features.pkl")

print("\n==============================")
print("MODEL FROZEN SUCCESSFULLY")
print("==============================")
print("Saved:")
print(" - models/sib_model.pkl")
print(" - models/sib_scaler.pkl")
print(" - models/sib_features.pkl")
print("==============================\n")
