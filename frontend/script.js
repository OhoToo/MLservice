const API_URL = window.location.port === "8000" ? window.location.origin : "http://localhost:8000";

const DEFAULT_VALUES = {
  person_gender: "male",
  person_education: "Bachelor",
  person_emp_exp: 0,
  person_home_ownership: "RENT",
  loan_intent: "PERSONAL",
  loan_int_rate: 0,
  cb_person_cred_hist_length: 0,
  credit_score: 650,
  previous_loan_defaults_on_file: "No",
};

const NUMERIC_FIELDS = new Set([
  "person_age",
  "person_income",
  "person_emp_exp",
  "loan_amnt",
  "loan_int_rate",
  "loan_percent_income",
  "cb_person_cred_hist_length",
  "credit_score",
]);

const INTEGER_FIELDS = new Set([
  "person_emp_exp",
  "credit_score",
]);

function showResult(elementId, data) {
  document.getElementById(elementId).textContent = JSON.stringify(data, null, 2);
}

async function handleResponse(response) {
  const data = await response.json();

  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join("; ")
      : data.detail;

    throw new Error(detail || "Request failed");
  }

  return data;
}

function parseNumber(value, fieldName) {
  const trimmedValue = String(value ?? "").trim();

  if (trimmedValue === "") {
    return DEFAULT_VALUES[fieldName] ?? null;
  }

  const parsed = Number(trimmedValue.replace(",", "."));

  if (Number.isNaN(parsed)) {
    throw new Error(`Некорректное число в поле: ${fieldName}`);
  }

  return INTEGER_FIELDS.has(fieldName) ? Math.round(parsed) : parsed;
}

function validateRecord(record) {
  if (record.person_age < 10 || record.person_age > 150) {
    throw new Error("Возраст должен быть от 10 до 150");
  }

  if (record.person_income <= 0) {
    throw new Error("Годовой доход должен быть больше 0");
  }

  if (record.loan_amnt <= 0) {
    throw new Error("Сумма кредита должна быть больше 0");
  }

  if (record.loan_percent_income < 0 || record.loan_percent_income > 1) {
    throw new Error("Доля кредита от дохода должна быть от 0 до 1");
  }

  if (record.loan_int_rate < 0 || record.loan_int_rate > 100) {
    throw new Error("Процентная ставка должна быть от 0 до 100");
  }

  if (record.credit_score < 0 || record.credit_score > 900) {
    throw new Error("Кредитный рейтинг должен быть от 0 до 900");
  }
}

function buildClientRecord(formData) {
  const record = {};

  for (const [key, value] of formData.entries()) {
    if (NUMERIC_FIELDS.has(key)) {
      record[key] = parseNumber(value, key);
    } else {
      record[key] = value || DEFAULT_VALUES[key];
    }
  }

  for (const [key, value] of Object.entries(DEFAULT_VALUES)) {
    if (record[key] === null || record[key] === undefined || record[key] === "") {
      record[key] = value;
    }
  }

  if (record.loan_percent_income === null) {
    record.loan_percent_income = Number((record.loan_amnt / record.person_income).toFixed(4));
  }

  validateRecord(record);
  return record;
}

document.getElementById("uploadModelBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("modelFile");

  if (!fileInput.files.length) {
    showResult("modelResult", { error: "Выберите .pkl файл" });
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const response = await fetch(`${API_URL}/upload-model`, {
      method: "POST",
      body: formData,
    });

    showResult("modelResult", await handleResponse(response));
  } catch (error) {
    showResult("modelResult", { error: error.message });
  }
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const form = document.getElementById("predictionForm");

  try {
    const record = buildClientRecord(new FormData(form));
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([record]),
    });

    showResult("predictionResult", await handleResponse(response));
  } catch (error) {
    showResult("predictionResult", { error: error.message });
  }
});

document.getElementById("csvPredictBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("csvFile");

  if (!fileInput.files.length) {
    showResult("csvResult", { error: "Выберите CSV файл" });
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const response = await fetch(`${API_URL}/predict-from-csv`, {
      method: "POST",
      body: formData,
    });
    const data = await handleResponse(response);
    const preview = {
      roc_auc: data.roc_auc,
      rows_count: data.data.length,
      first_rows: data.data.slice(0, 5),
    };

    showResult("csvResult", preview);
  } catch (error) {
    showResult("csvResult", { error: error.message });
  }
});
