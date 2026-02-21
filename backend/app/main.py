from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.prediction_routes import router as prediction_router
from app.api.hdf5_routes import router as hdf5_router

app = FastAPI(title="SIB Battery Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(hdf5_router)
