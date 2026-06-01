from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model_service import predict_csv_dataframe, predict_dataframe, save_uploaded_model
from app.schemas import ClientData

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Mortgage Approval ML Service",
    description="API for mortgage approval prediction using a scikit-learn model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)) -> dict[str, str]:
    return await save_uploaded_model(file)


@app.post("/predict")
def predict(records: list[ClientData]) -> list[dict]:
    if not records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one record is required",
        )

    input_df = pd.DataFrame([record.model_dump() for record in records])
    result_df = predict_dataframe(input_df)
    return result_df.to_dict(orient="records")


@app.post("/predict-from-csv")
async def predict_from_csv(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are allowed",
        )

    content = await file.read()
    try:
        input_df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CSV file: {exc}",
        ) from exc

    return predict_csv_dataframe(input_df)