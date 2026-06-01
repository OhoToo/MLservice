const API_URL = window.location.port === "8000"
  ? window.location.origin
  : "http://localhost:8000";

function showResult(elementId, data) {
  document.getElementById(elementId).textContent = JSON.stringify(data, null, 2);
}

async function handleResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
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
  const formData = new FormData(form);
  const numericFields = new Set([
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
  ]);

  const record = {};
  for (const [key, value] of formData.entries()) {
    record[key] = numericFields.has(key) ? Number(value) : value;
  }

  try {
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