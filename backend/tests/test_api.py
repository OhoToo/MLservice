from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_without_model_returns_error(tmp_path, monkeypatch) -> None:
    import app.model_service as model_service

    monkeypatch.setattr(model_service, "_model_bundle", None)
    monkeypatch.setattr(model_service, "MODEL_PATH", tmp_path / "missing_model.pkl")

    payload = [
        {
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
    ]
    response = client.post("/predict", json=payload)
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
