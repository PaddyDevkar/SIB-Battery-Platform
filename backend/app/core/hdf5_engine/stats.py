import numpy as np


def compute_dataset_stats(data):
    data = np.array(data)

    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "size": int(data.size)
    }
