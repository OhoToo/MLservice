# Mortgage Approval ML Service

Лабораторная работа №10 по дисциплине «Программирование на Python».

Тема: разработка ML-сервиса для предсказания одобрения ипотеки.

## Описание

Проект представляет собой веб-приложение для предсказания статуса ипотечной заявки. Сервис принимает данные клиента и возвращает прогноз:

- `approved` — заявка одобрена;
- `rejected` — заявка отклонена.

Backend реализован на FastAPI. Модель машинного обучения обучается с помощью scikit-learn и сохраняется в формате `.pkl`. Frontend реализован как простой HTML/CSS/JavaScript-интерфейс.

## Датасет

Используется файл `data/loan_data.csv`.

Признаки:

| Колонка | Описание |
|---|---|
| `person_age` | возраст клиента |
| `person_gender` | пол клиента |
| `person_education` | уровень образования |
| `person_income` | годовой доход |
| `person_emp_exp` | опыт работы |
| `person_home_ownership` | статус владения жильём |
| `loan_amnt` | запрашиваемая сумма кредита |
| `loan_intent` | цель кредита |
| `loan_int_rate` | процентная ставка |
| `loan_percent_income` | доля кредита от дохода |
| `cb_person_cred_hist_length` | длина кредитной истории |
| `credit_score` | кредитный рейтинг |
| `previous_loan_defaults_on_file` | наличие предыдущих дефолтов |
| `loan_status` | целевая переменная: `1 = approved`, `0 = rejected` |

## ML pipeline

В `ml/train_model.py` реализованы:

1. Загрузка датасета.
2. Очистка данных.
3. Удаление дубликатов.
4. Обработка выбросов.
5. Разделение на train/test.
6. Кодирование категориальных признаков через `OneHotEncoder`.
7. Масштабирование числовых признаков через `StandardScaler`.
8. Feature selection через `SelectPercentile`.
9. Обучение и сравнение моделей.
10. Выбор лучшей модели по ROC-AUC.
11. Сохранение модели в `.pkl`.

Сравниваются модели:

- Logistic Regression;
- Random Forest;
- Gradient Boosting.

## API

### GET `/health`

Проверка состояния сервиса.

Ответ:

```json
{
  "status": "ok"
}
```

### POST `/upload-model`

Загрузка модели в формате `.pkl`.

Формат запроса: `multipart/form-data`.

Поле: `file`.

Ответ:

```json
{
  "status": "success",
  "message": "Model uploaded successfully"
}
```

### POST `/predict`

Предсказание по одной или нескольким JSON-записям.

Пример запроса:

```json
[
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
    "previous_loan_defaults_on_file": "No"
  }
]
```

Пример ответа:

```json
[
  {
    "person_age": 25,
    "person_gender": "female",
    "predicted_loan_status": "approved",
    "approval_probability": 0.9132
  }
]
```

### POST `/predict-from-csv`

Предсказание по CSV-файлу.

Если в CSV есть колонка `loan_status`, сервис дополнительно возвращает `roc_auc`.

Ответ:

```json
{
  "roc_auc": 0.94,
  "data": [
    {
      "person_age": 25,
      "predicted_loan_status": "approved",
      "approval_probability": 0.9132
    }
  ]
}
```

## Запуск ML-обучения

Из корня проекта:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python ml/train_model.py
```

После обучения модель появится здесь:

```text
ml/models/best_model.pkl
```

Для работы backend её можно загрузить через `/upload-model` или скопировать:

```bash
cp ml/models/best_model.pkl backend/app/models/model.pkl
```

## Запуск backend

```bash
cd backend
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Запуск frontend

Открыть файл:

```text
frontend/index.html
```

Backend должен быть запущен на:

```text
http://localhost:8000
```

## Тестирование

```bash
cd backend
pytest
```

## Линтинг backend

```bash
cd backend
ruff check .
```

## Линтинг frontend

```bash
cd frontend
npm install
npm run lint
```

## CI Sourcecraft

CI-сценарий находится в файле:

```text
.sourcecraft/ci.yml
```

Он выполняет:

1. Линтинг backend через `ruff`.
2. Тестирование backend через `pytest`.
3. Линтинг frontend через `eslint`.

## Распределение ролей

### ML Engineer

- Предобработка данных.
- Обучение моделей.
- Сравнение ROC-AUC.
- Сохранение лучшей модели.

### Backend Developer

- FastAPI API.
- Загрузка модели.
- Предсказания по JSON и CSV.
- Обработка ошибок.
- Тесты.

### Frontend Developer

- HTML/CSS/JS-интерфейс.
- Интеграция с API.
- Отображение результатов.

## Дополнительно реализовано

- CORS для frontend.
- Валидация входных данных через Pydantic.
- Обработка ошибок при отсутствии модели.
- Возврат вероятности одобрения.
- CI-сценарий.
- Линтинг backend и frontend.
