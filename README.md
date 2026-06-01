# Mortgage Approval ML Service

Лабораторная работа №10 по дисциплине «Программирование на Python».

Тема: разработка ML-сервиса для предсказания одобрения ипотеки.

## Описание проекта

Проект представляет собой веб-приложение для предсказания статуса ипотечной заявки.

Сервис принимает данные клиента и возвращает прогноз:

- `approved` — заявка одобрена;
- `rejected` — заявка отклонена.

Backend реализован на FastAPI. ML-модель обучается с помощью scikit-learn и сохраняется в формате `.pkl`. Frontend реализован как простой HTML/CSS/JavaScript-интерфейс и отдаётся самим FastAPI-приложением.

## Структура проекта

```text
mortgage-approval-service/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model_service.py
│   │   ├── schemas.py
│   │   └── models/
│   │       └── model.pkl
│   ├── tests/
│   │   └── test_api.py
│   ├── pyproject.toml
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   ├── package.json
│   ├── eslint.config.js
│   └── scripts/
│       └── check-static-files.js
├── ml/
│   ├── train_model.py
│   └── models/
│       └── best_model.pkl
├── data/
│   ├── loan_data.csv
│   └── test_clients.csv
├── .sourcecraft/
│   └── ci.yaml
├── .gitignore
└── README.md
```

## Датасет

Используется файл:

```text
data/loan_data.csv
```

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

В файле `ml/train_model.py` реализованы:

1. Загрузка датасета.
2. Проверка и очистка данных.
3. Удаление дубликатов.
4. Обработка выбросов.
5. Разделение на train/test.
6. Кодирование категориальных признаков через `OneHotEncoder`.
7. Масштабирование числовых признаков через `StandardScaler`.
8. Feature selection через `SelectPercentile`.
9. Обучение нескольких моделей.
10. Сравнение моделей по ROC-AUC.
11. Сохранение лучшей модели в `.pkl`.

Сравниваются модели:

- Logistic Regression;
- Random Forest;
- Gradient Boosting.

Лучшая модель сохраняется в:

```text
ml/models/best_model.pkl
```

Для удобства запуска backend готовая модель также лежит в:

```text
backend/app/models/model.pkl
```

## API

Пользовательский интерфейс доступен на главной странице `http://localhost:8000`. Swagger-документация для разработчика доступна по адресу `http://localhost:8000/docs`.


### GET `/health`

Проверка состояния сервиса.

Пример ответа:

```json
{
  "status": "ok"
}
```

### POST `/upload-model`

Загрузка модели в формате `.pkl`.

Формат запроса:

```text
multipart/form-data
```

Поле:

```text
file
```

Пример ответа:

```json
{
  "status": "success",
  "message": "Model uploaded successfully"
}
```

### POST `/predict`

Предсказание по одной или нескольким JSON-записям.

Обязательные поля для ручного ввода:

- `person_age`;
- `person_gender`;
- `person_income`;
- `loan_amnt`.

Остальные поля можно не передавать. В этом случае сервис использует значения по умолчанию:

| Поле | Значение по умолчанию |
|---|---|
| `person_education` | `Bachelor` |
| `person_emp_exp` | `0` |
| `person_home_ownership` | `RENT` |
| `loan_intent` | `PERSONAL` |
| `loan_int_rate` | `0` |
| `loan_percent_income` | рассчитывается как `loan_amnt / person_income` |
| `cb_person_cred_hist_length` | `0` |
| `credit_score` | `650` |
| `previous_loan_defaults_on_file` | `No` |

Frontend автоматически заменяет запятую на точку в дробных числах. Например, `10,5` будет обработано как `10.5`.

Пример полного запроса:

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


Пример минимального запроса:

```json
[
  {
    "person_age": 35,
    "person_gender": "male",
    "person_income": 85000,
    "loan_amnt": 20000,
    "loan_int_rate": "10,5"
  }
]
```

Пример ответа:

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
    "previous_loan_defaults_on_file": "No",
    "predicted_loan_status": "approved",
    "approval_probability": 0.9132
  }
]
```

### POST `/predict-from-csv`

Предсказание по CSV-файлу.

Если в CSV есть колонка `loan_status`, сервис дополнительно возвращает `roc_auc`.

Для быстрой проверки можно загрузить файл `data/test_clients.csv`. В нём нет `loan_status`, поэтому `roc_auc` будет `null`, но сервис вернёт предсказания для всех строк.

Пример ответа:

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

## Запуск на Windows PowerShell

Из корня проекта:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: -r backend\requirements.txt
```

Обучение модели:

```powershell
python ml\train_model.py
copy ml\models\best_model.pkl backend\app\models\model.pkl
```

Запуск backend:

```powershell
cd backend
uvicorn app.main:app --reload
```

После запуска доступны:

```text
http://localhost:8000        пользовательский интерфейс
http://localhost:8000/docs   Swagger-документация API
http://localhost:8000/health проверка состояния сервиса
```

## Запуск на Linux/macOS

Из корня проекта:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: -r backend/requirements.txt
python ml/train_model.py
cp ml/models/best_model.pkl backend/app/models/model.pkl
cd backend
uvicorn app.main:app --reload
```

После запуска открыть:

```text
http://localhost:8000
```

## Тестирование backend

```bash
cd backend
pytest
```

В тестах проверяются:

- успешный `/health`;
- успешный `/predict`;
- успешный `/predict-from-csv`;
- успешный `/upload-model`;
- ошибка при отсутствии модели;
- ошибка при загрузке файла модели с неправильным расширением;
- ошибка при загрузке CSV с неправильным расширением.

## Линтинг backend

```bash
cd backend
ruff check .
```

## Линтинг и сборка frontend

```bash
cd frontend
npm install
npm run lint
npm run build
```

Frontend статический, поэтому команда `npm run build` выполняет проверку наличия основных файлов интерфейса.

## CI SourceCraft

CI-сценарий находится в файле:

```text
.sourcecraft/ci.yaml
```

Сценарий запускается на `push` и `pull_request` в ветки `main` и `master`.

Он выполняет:

1. Установку backend-зависимостей.
2. Линтинг backend через `ruff`.
3. Тестирование backend через `pytest`.
4. Установку frontend-зависимостей.
5. Линтинг frontend через `eslint`.
6. Проверку сборки frontend через `npm run build`.

## Коммит проекта

Перед отправкой в репозиторий нужно убедиться, что в коммит не попали виртуальное окружение и временные файлы.

Проверка:

```bash
git status
```

Коммит:

```bash
git add .
git commit -m "Complete mortgage approval ML service"
git push
```

Файл `.gitignore` исключает `.venv`, `__pycache__`, `.pytest_cache`, `node_modules` и другой мусор. При этом обученные `.pkl`-модели оставлены в проекте, потому что лабораторная требует наличие сохранённой модели.

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

- Пользовательский интерфейс открывается сразу на `http://localhost:8000`.
- Swagger-документация доступна на `http://localhost:8000/docs`.
- CORS для frontend.
- Валидация входных данных через Pydantic.
- Обработка ошибок при отсутствии модели.
- Возврат вероятности одобрения.
- CI-сценарий SourceCraft.
- Линтинг backend и frontend.
- Backend API-тесты, включая успешные запросы.
