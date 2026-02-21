import h5py
import pandas as pd


def load_signal(file_path: str):
    """
    Loads raw signal data from HDF5 file.

    Returns:
        pandas DataFrame with:
        - Current(A)
        - Voltage(V)
        - Test_Time
        - Cycle_Index
    """

    try:
        with h5py.File(file_path, "r") as f:

            # Assumes standard structure used in training
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

    except Exception as e:
        print("❌ Error loading HDF5 file:", e)
        return None
