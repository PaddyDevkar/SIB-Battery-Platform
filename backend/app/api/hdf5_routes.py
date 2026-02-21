from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile
import os
import h5py
import numpy as np

router = APIRouter(prefix="/hdf5", tags=["HDF5 Explorer"])


# ==========================================================
# 🔹 1️⃣ Build Recursive HDF5 Tree
# ==========================================================

def build_tree(hdf, path="/"):
    nodes = []

    try:
        for key in hdf[path]:
            full_path = path.rstrip("/") + "/" + key

            obj = hdf[full_path]

            if isinstance(obj, h5py.Group):
                nodes.append({
                    "name": key,
                    "path": full_path,
                    "type": "group",
                    "children": build_tree(hdf, full_path)
                })

            elif isinstance(obj, h5py.Dataset):
                nodes.append({
                    "name": key,
                    "path": full_path,
                    "type": "dataset",
                    "shape": obj.shape,
                    "dtype": str(obj.dtype)
                })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return nodes


@router.post("/structure")
async def explore_structure(file: UploadFile = File(...)):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        with h5py.File(temp_path, "r") as hdf:
            tree = build_tree(hdf)

        return {
            "status": "success",
            "tree": tree
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================================
# 🔹 2️⃣ Dataset Utilities
# ==========================================================

def slice_dataset(dataset, start=None, end=None, stride=1):
    total_length = dataset.shape[0]

    start = 0 if start is None else max(0, start)
    end = total_length if end is None or end == 0 else min(end, total_length)
    stride = max(1, stride)

    return dataset[start:end:stride]


def compute_stats(data):
    arr = np.array(data)

    if arr.size == 0:
        return {}

    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "size": int(arr.size)
    }


def smart_downsample(data, max_points=3000):
    arr = np.array(data)

    if arr.ndim != 1:
        return arr.tolist()  # For matrix datasets, don't downsample

    if len(arr) <= max_points:
        return arr.tolist()

    factor = max(1, len(arr) // max_points)
    return arr[::factor].tolist()


# ==========================================================
# 🔹 3️⃣ Load Dataset (Safe + Optimized)
# ==========================================================

@router.post("/dataset")
async def get_dataset(
    file: UploadFile = File(...),
    dataset_path: str = Form(...),
    start: int = Form(None),
    end: int = Form(None),
    stride: int = Form(1)
):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "r") as hdf:

            if dataset_path not in hdf:
                return {"detail": f"Dataset path '{dataset_path}' not found."}

            obj = hdf[dataset_path]

            # 🔥 Ensure it's actually a dataset
            if not isinstance(obj, h5py.Dataset):
                return {"detail": f"Path '{dataset_path}' is not a dataset."}

            shape = obj.shape
            dtype = str(obj.dtype)

            raw_data = slice_dataset(obj, start, end, stride)
            stats = compute_stats(raw_data)
            downsampled = smart_downsample(raw_data)

        return {
            "status": "success",
            "dataset_path": dataset_path,
            "shape": shape,
            "dtype": dtype,
            "data": downsampled,
            "stats": stats
        }

    finally:
        os.remove(tmp_path)
