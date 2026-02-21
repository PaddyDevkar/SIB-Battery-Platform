from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.prediction_service import run_prediction

router = APIRouter()

UPLOAD_DIR = "storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/predict")
async def predict_battery(file: UploadFile = File(...)):
    """
    Upload HDF5 file and get prediction.
    """

    if not file.filename.endswith(".hdf5"):
        return {"error": "Only .hdf5 files are supported."}

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_prediction(file_path)

    return result
