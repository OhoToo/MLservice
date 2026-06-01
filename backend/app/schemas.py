from __future__ import annotations

from pydantic import BaseModel, Field


class ClientData(BaseModel):
    person_age: float = Field(..., ge=18, le=100)
    person_gender: str
    person_education: str
    person_income: float = Field(..., ge=0)
    person_emp_exp: int = Field(..., ge=0)
    person_home_ownership: str
    loan_amnt: float = Field(..., ge=0)
    loan_intent: str
    loan_int_rate: float = Field(..., ge=0, le=100)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_cred_hist_length: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=900)
    previous_loan_defaults_on_file: str

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }
