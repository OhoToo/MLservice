from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException, UploadFile, status
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
TARGET_COLUMN = "loan_status"
APPROVED_LABEL = 1
REJECTED_LABEL = 0

NUMERIC_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

DEFAULT_FEATURE_VALUES: dict[str, Any] = {
    "person_gender": "male",
    "person_education": "Bachelor",
    "person_emp_exp": 0,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_int_rate": 0,
    "cb_person_cred_hist_length": 0,
    "credit_score": 650,
    "previous_loan_defaults_on_file": "No",
}

_model_bundle: dict[str, Any] | None = None


def _extract_model(bundle_or_model: Any) -> Any:
    if isinstance(bundle_or_model, dict) and "model" in bundle_or_model:
        return bundle_or_model["model"]
    return bundle_or_model


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    global _model_bundle

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model is not loaded",
        )

    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "model" in loaded:
        _model_bundle = loaded
    else:
        _model_bundle = {"model": loaded, "target_column": TARGET_COLUMN}
    return _model_bundle


def get_model_bundle() -> dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if MODEL_PATH.exists():
            return load_model(MODEL_PATH)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model is not loaded",
        )
    return _model_bundle


async def save_uploaded_model(file: UploadFile) -> dict[str, str]:
    if not file.filename or not file.filename.endswith(".pkl"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pkl model files are allowed",
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    MODEL_PATH.write_bytes(content)

    try:
        load_model(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        MODEL_PATH.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model file: {exc}",
        ) from exc

    return {"status": "success", "message": "Model uploaded successfully"}


def _normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            )
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _fill_optional_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for column in feature_columns:
        if column not in df.columns and column in DEFAULT_FEATURE_VALUES:
            df[column] = DEFAULT_FEATURE_VALUES[column]

    if "loan_percent_income" not in df.columns and {"loan_amnt", "person_income"}.issubset(df.columns):
        df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]

    for column, default_value in DEFAULT_FEATURE_VALUES.items():
        if column in df.columns:
            df[column] = df[column].fillna(default_value)

    if "loan_percent_income" in df.columns and {"loan_amnt", "person_income"}.issubset(df.columns):
        calculated_ratio = df["loan_amnt"] / df["person_income"]
        df["loan_percent_income"] = df["loan_percent_income"].fillna(calculated_ratio)

    return df


def normalize_input_frame(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    target_column = bundle.get("target_column", TARGET_COLUMN)

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    feature_columns = bundle.get("feature_columns")
    if feature_columns:
        df = _normalize_numeric_columns(df)
        df = _fill_optional_columns(df, feature_columns)

        missing_columns = [column for column in feature_columns if column not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required columns: {missing_columns}",
            )
        df = df[feature_columns]

    return df


def predict_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    bundle = get_model_bundle()
    model = _extract_model(bundle)
    features_df = normalize_input_frame(input_df, bundle)

    try:
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)[:, 1]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed: {exc}",
        ) from exc

    result_df = input_df.copy()
    result_df["predicted_loan_status"] = [
        "approved" if int(value) == APPROVED_LABEL else "rejected" for value in predictions
    ]
    result_df["approval_probability"] = probabilities.round(4)
    return result_df


def predict_csv_dataframe(input_df: pd.DataFrame) -> dict[str, Any]:
    bundle = get_model_bundle()
    model = _extract_model(bundle)
    target_column = bundle.get("target_column", TARGET_COLUMN)

    y_true = None
    if target_column in input_df.columns:
        y_true = input_df[target_column].astype(int)

    result_df = predict_dataframe(input_df)

    roc_auc = None
    if y_true is not None:
        features_df = normalize_input_frame(input_df, bundle)
        probabilities = model.predict_proba(features_df)[:, 1]
        roc_auc = round(float(roc_auc_score(y_true, probabilities)), 4)

    return {
        "roc_auc": roc_auc,
        "data": result_df.to_dict(orient="records"),
    }
