import numpy as np


def slice_dataset(dataset, start=None, end=None, stride=1):

    if start is None:
        start = 0

    if end is None:
        end = dataset.shape[0]

    data = dataset[start:end:stride]

    return data


def smart_downsample(data, max_points=2000):
    data = np.array(data)

    if len(data) <= max_points:
        return data.tolist()

    factor = len(data) // max_points
    return data[::factor].tolist()
