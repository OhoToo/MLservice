from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ClientData(BaseModel):
    person_age: float = Field(..., ge=10, le=150)
    person_gender: str = Field(default="male")
    person_education: str = Field(default="Bachelor")
    person_income: float = Field(..., gt=0)
    person_emp_exp: int = Field(default=0, ge=0, le=80)
    person_home_ownership: str = Field(default="RENT")
    loan_amnt: float = Field(..., gt=0)
    loan_intent: str = Field(default="PERSONAL")
    loan_int_rate: float = Field(default=0, ge=0, le=100)
    loan_percent_income: float | None = Field(default=None, ge=0, le=1)
    cb_person_cred_hist_length: float = Field(default=0, ge=0, le=100)
    credit_score: int = Field(default=650, ge=0, le=900)
    previous_loan_defaults_on_file: str = Field(default="No")

    @field_validator(
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
        mode="before",
    )
    @classmethod
    def normalize_decimal_separator(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped_value = value.strip()
            if stripped_value == "":
                return None
            return stripped_value.replace(",", ".")
        return value

    @model_validator(mode="after")
    def fill_calculated_fields(self) -> ClientData:
        if self.loan_percent_income is None:
            self.loan_percent_income = round(self.loan_amnt / self.person_income, 4)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 35,
                "person_gender": "male",
                "person_income": 85000,
                "loan_amnt": 20000,
                "loan_int_rate": "10,5",
                "loan_percent_income": "0,24",
            }
        }
    }
