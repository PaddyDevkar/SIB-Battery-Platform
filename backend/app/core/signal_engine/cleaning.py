import numpy as np


def clean_signal(df):
    required = ["Voltage(V)", "Current(A)", "Test_Time"]

    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=required)
    df = df.sort_values("Test_Time").reset_index(drop=True)

    return df


def regenerate_cycles(df):
    """
    Normalize existing Cycle_Index into sequential cycle numbers.
    """

    if "Cycle_Index" not in df.columns:
        return df

    df = df.copy()

    # Map unique cycle IDs to sequential integers
    unique_cycles = sorted(df["Cycle_Index"].unique())
    cycle_map = {cycle_id: idx for idx, cycle_id in enumerate(unique_cycles)}

    df["Cycle_Index"] = df["Cycle_Index"].map(cycle_map)

    return df
