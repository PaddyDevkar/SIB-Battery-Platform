import numpy as np


def generate_cycle_summary(df):
    """
    Convert time-series dataframe into cycle-level summary.
    """

    if "Cycle_Index" not in df.columns:
        return None

    if df["Cycle_Index"].nunique() < 5:
        return None

    cycle_df = df.groupby("Cycle_Index").agg({
        "Charge_Capacity(Ah)": "max",
        "Discharge_Capacity(Ah)": "max",
        "Charge_Energy": "sum",
        "Discharge_Energy": "sum",
        "Voltage(V)": "mean",
        "Current(A)": "mean"
    }).reset_index()

    cycle_df.rename(columns={
        "Charge_Capacity(Ah)": "charge_capacity",
        "Discharge_Capacity(Ah)": "discharge_capacity",
        "Charge_Energy": "charge_energy",
        "Discharge_Energy": "discharge_energy",
        "Voltage(V)": "voltage_mean",
        "Current(A)": "current_mean"
    }, inplace=True)

    # Voltage std
    voltage_std = (
        df.groupby("Cycle_Index")["Voltage(V)"]
        .std()
        .fillna(0)
        .values
    )

    cycle_df["voltage_std"] = voltage_std

    # Coulombic Efficiency
    cycle_df["CE"] = np.where(
        cycle_df["charge_capacity"] > 0,
        cycle_df["discharge_capacity"] /
        cycle_df["charge_capacity"],
        np.nan
    )

    # Energy loss
    cycle_df["energy_loss"] = (
        cycle_df["charge_energy"] -
        cycle_df["discharge_energy"]
    )

    # Voltage hysteresis proxy
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

    # Capacity retention
    if cycle_df["discharge_capacity"].iloc[0] > 0:
        cycle_df["capacity_retention"] = (
            cycle_df["discharge_capacity"] /
            cycle_df["discharge_capacity"].iloc[0]
        )
    else:
        cycle_df["capacity_retention"] = np.nan

    return cycle_df
