from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_RECORD = {
    "person_age": 25,
    "person_gender": "female",
    "person_education": "Bachelor",
    "person_income": 75000,
    "person_emp_exp": 3,
    "person_home_ownership": "RENT",
    "loan_amnt": 12000,
    "loan_intent": "EDUCATION",
    "loan_int_rate": 10.5,
    "loan_percent_income": 0.16,
    "cb_person_cred_hist_length": 4,
    "credit_score": 680,
    "previous_loan_defaults_on_file": "No",
}

PARTIAL_RECORD = {
    "person_age": 35,
    "person_gender": "male",
    "person_income": 85000,
    "loan_amnt": 20000,
    "loan_int_rate": "10,5",
}


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success() -> None:
    response = client.post("/predict", json=[VALID_RECORD])
    data = response.json()

    assert response.status_code == 200
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["predicted_loan_status"] in {"approved", "rejected"}
    assert 0 <= data[0]["approval_probability"] <= 1


def test_predict_success_with_optional_defaults_and_comma_decimal() -> None:
    response = client.post("/predict", json=[PARTIAL_RECORD])
    data = response.json()

    assert response.status_code == 200
    assert data[0]["person_education"] == "Bachelor"
    assert data[0]["person_home_ownership"] == "RENT"
    assert data[0]["cb_person_cred_hist_length"] == 0
    assert data[0]["credit_score"] == 650
    assert data[0]["loan_int_rate"] == 10.5
    assert data[0]["predicted_loan_status"] in {"approved", "rejected"}


def test_predict_from_csv_success() -> None:
    csv_content = (
        ",".join([*VALID_RECORD.keys(), "loan_status"])
        + "\n"
        + ",".join([str(value) for value in VALID_RECORD.values()])
        + ",1\n"
        + ",".join([str(value) for value in {**VALID_RECORD, "loan_status": 0}.values()])
        + "\n"
    )

    response = client.post(
        "/predict-from-csv",
        files={"file": ("records.csv", csv_content.encode(), "text/csv")},
    )
    data = response.json()

    assert response.status_code == 200
    assert "roc_auc" in data
    assert len(data["data"]) == 2
    assert data["data"][0]["predicted_loan_status"] in {"approved", "rejected"}


def test_predict_from_csv_success_with_optional_defaults() -> None:
    csv_content = "person_age,person_gender,person_income,loan_amnt,loan_int_rate\n35,male,85000,20000,10.5\n"

    response = client.post(
        "/predict-from-csv",
        files={"file": ("records.csv", csv_content.encode(), "text/csv")},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["roc_auc"] is None
    assert len(data["data"]) == 1
    assert data["data"][0]["predicted_loan_status"] in {"approved", "rejected"}


def test_upload_model_success() -> None:
    model_path = Path(__file__).resolve().parents[1] / "app" / "models" / "model.pkl"

    with model_path.open("rb") as model_file:
        response = client.post(
            "/upload-model",
            files={"file": ("model.pkl", model_file.read(), "application/octet-stream")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_predict_without_model_returns_error(tmp_path, monkeypatch) -> None:
    import app.model_service as model_service

    monkeypatch.setattr(model_service, "_model_bundle", None)
    monkeypatch.setattr(model_service, "MODEL_PATH", tmp_path / "missing_model.pkl")

    response = client.post("/predict", json=[VALID_RECORD])

    assert response.status_code == 400
    assert response.json()["detail"] == "Model is not loaded"


def test_upload_model_rejects_wrong_extension() -> None:
    response = client.post(
        "/upload-model",
        files={"file": ("model.txt", b"not a pickle", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .pkl" in response.json()["detail"]


def test_predict_from_csv_rejects_wrong_extension() -> None:
    response = client.post(
        "/predict-from-csv",
        files={"file": ("data.txt", b"a,b\n1,2", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .csv" in response.json()["detail"]
