import numpy as np


def compute_capacity_energy(df):
    """
    Compute:
    - dAh
    - dWh
    - Charge_Capacity(Ah)
    - Discharge_Capacity(Ah)
    - Charge_Energy
    - Discharge_Energy
    """

    df = df.copy()

    # Time difference
    df["dt"] = df["Test_Time"].diff().fillna(0)

    # Capacity increment
    df["dAh"] = df["Current(A)"] * df["dt"] / 3600.0

    # Energy increment
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
