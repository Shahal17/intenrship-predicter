from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_dataset() -> Path:
    csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(
            "No CSV dataset found in data/. Run `python src/download_data.py` first "
            "or manually place the Kaggle CSV inside data/."
        )
    return csv_files[0]


def find_target_column(columns: list[str]) -> str:
    normalized = {normalize_name(column): column for column in columns}
    preferred = [
        "placementstatus",
        "placedstatus",
        "isplaced",
        "placed",
        "placement",
        "status",
    ]

    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]

    for column in columns:
        key = normalize_name(column)
        if "placement" in key and ("status" in key or "placed" in key):
            return column

    raise ValueError(
        "Could not automatically find the placement target column. "
        f"Available columns: {columns}"
    )


def encode_binary_target(series: pd.Series) -> tuple[pd.Series, dict[int, str]]:
    clean = series.astype(str).str.strip()
    unique = list(pd.unique(clean))
    if len(unique) != 2:
        raise ValueError(
            f"Expected a binary placement target, but found {len(unique)} classes: {unique}"
        )

    def label_kind(label: str) -> str | None:
        value = normalize_name(label)
        negative_exact = {"0", "no", "false", "unplaced", "notplaced", "notselected"}
        positive_exact = {"1", "yes", "true", "placed", "selected"}

        if value in negative_exact or "unplaced" in value or "notplaced" in value:
            return "negative"
        if value in positive_exact or ("placed" in value and "not" not in value):
            return "positive"
        return None

    positive = [label for label in unique if label_kind(label) == "positive"]
    negative = [label for label in unique if label_kind(label) == "negative"]

    if len(positive) == 1 and len(negative) == 1:
        mapping = {negative[0]: 0, positive[0]: 1}
    else:
        ordered = sorted(unique)
        mapping = {ordered[0]: 0, ordered[1]: 1}
        print(
            "Warning: target labels were ambiguous. Using alphabetical binary mapping: "
            f"{mapping}"
        )

    encoded = clean.map(mapping).astype(int)
    inverse = {encoded_value: original for original, encoded_value in mapping.items()}
    return encoded, inverse


def should_drop_feature(column: str) -> bool:
    key = normalize_name(column)

    if key.startswith("unnamed"):
        return True
    if key in {"studentid", "studentname", "name", "id", "rollno", "rollnumber"}:
        return True
    if key.endswith("studentid") or key.endswith("rollno"):
        return True

    # These are normally known only after a placement outcome and can leak the target.
    leakage_terms = ("salary", "package", "company", "offerstatus", "offeredcompany")
    return any(term in key for term in leakage_terms)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns


def make_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def build_feature_schema(
    X: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> list[dict]:
    schema: list[dict] = []

    for column in X.columns:
        if column in numeric_columns:
            numeric = pd.to_numeric(X[column], errors="coerce")
            non_null = numeric.dropna()
            schema.append(
                {
                    "name": column,
                    "type": "numeric",
                    "default": float(non_null.median()) if not non_null.empty else 0.0,
                    "min": float(non_null.min()) if not non_null.empty else 0.0,
                    "max": float(non_null.max()) if not non_null.empty else 100.0,
                }
            )
        elif column in categorical_columns:
            values = X[column].dropna().astype(str).value_counts().index.tolist()[:50]
            schema.append(
                {
                    "name": column,
                    "type": "categorical",
                    "choices": values,
                    "default": values[0] if values else "Unknown",
                }
            )

    return schema


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = [str(column).strip() for column in df.columns]

    target_column = find_target_column(df.columns.tolist())
    df = df.dropna(subset=[target_column]).copy()
    y, target_mapping = encode_binary_target(df[target_column])

    raw_features = df.drop(columns=[target_column])
    dropped_columns = [
        column for column in raw_features.columns if should_drop_feature(column)
    ]
    X = raw_features.drop(columns=dropped_columns, errors="ignore").copy()

    # Remove constant columns because they add no predictive information.
    constant_columns = [column for column in X.columns if X[column].nunique(dropna=False) <= 1]
    if constant_columns:
        X = X.drop(columns=constant_columns)
        dropped_columns.extend(constant_columns)

    if X.empty:
        raise ValueError("No usable input features remain after preprocessing checks.")

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metrics_rows: list[dict] = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in make_models().items():
        print(f"Training {model_name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        metrics_rows.append(metrics)
        trained_pipelines[model_name] = pipeline
        print(metrics)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["f1_score", "accuracy"], ascending=False
    )
    metrics_path = RESULTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_pipeline = trained_pipelines[best_model_name]
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    # Permutation importance works at the original input-column level and is model-agnostic.
    sample_size = min(2000, len(X_test))
    X_importance = X_test.sample(sample_size, random_state=RANDOM_STATE)
    y_importance = y_test.loc[X_importance.index]
    importance = permutation_importance(
        best_pipeline,
        X_importance,
        y_importance,
        scoring="f1",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    metadata = {
        "dataset_file": dataset_path.name,
        "target_column": target_column,
        "target_mapping": {str(key): value for key, value in target_mapping.items()},
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": dropped_columns,
        "best_model": best_model_name,
        "selection_metric": "f1_score",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "feature_schema": build_feature_schema(X, numeric_columns, categorical_columns),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
