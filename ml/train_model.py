from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "loan_data.csv"
MODEL_PATH = ROOT_DIR / "ml" / "models" / "best_model.pkl"
TARGET_COLUMN = "loan_status"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning: remove duplicates and unrealistic outliers."""
    df = df.copy()
    df = df.drop_duplicates()

    numeric_rules = {
        "person_age": (18, 100),
        "person_income": (0, df["person_income"].quantile(0.99)),
        "person_emp_exp": (0, 80),
        "loan_amnt": (0, df["loan_amnt"].quantile(0.99)),
        "loan_int_rate": (0, 100),
        "loan_percent_income": (0, 1),
        "cb_person_cred_hist_length": (0, 80),
        "credit_score": (300, 900),
    }

    for column, (min_value, max_value) in numeric_rules.items():
        if column in df.columns:
            df = df[(df[column] >= min_value) & (df[column] <= max_value)]

    return df.reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("feature_selection", SelectPercentile(score_func=f_classif, percentile=80)),
            ("classifier", model),
        ]
    )


def evaluate_models(X_train, X_test, y_train, y_test) -> tuple[str, Pipeline, dict[str, float]]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    scores: dict[str, float] = {}
    best_name = ""
    best_score = -1.0
    best_pipeline: Pipeline | None = None

    for name, model in models.items():
        pipeline = build_pipeline(model, X_train)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, probabilities)
        scores[name] = score
        print(f"{name} ROC-AUC: {score:.4f}")

        if score > best_score:
            best_name = name
            best_score = score
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("Failed to train models")

    y_pred = best_pipeline.predict(X_test)
    print("\nBest model:", best_name)
    print("Best ROC-AUC:", round(best_score, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    return best_name, best_pipeline, scores


def main() -> None:
    df = load_data()
    df = clean_data(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_name, best_pipeline, scores = evaluate_models(X_train, X_test, y_train, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_pipeline,
            "model_name": best_name,
            "roc_auc_scores": scores,
            "target_column": TARGET_COLUMN,
            "positive_label": 1,
            "negative_label": 0,
            "feature_columns": X.columns.tolist(),
        },
        MODEL_PATH,
    )
    print(f"\nSaved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
