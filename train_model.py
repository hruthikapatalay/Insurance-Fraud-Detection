import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "fraud_reported"

FEATURES = [
    "months_as_customer",
    "age",
    "policy_annual_premium",
    "total_claim_amount",
    "incident_severity",
    "collision_type",
    "witnesses",
    "police_report_available",
    "incident_hour_of_the_day",
]


def resolve_dataset_path() -> Path:
    candidates = [Path("frauddetection.csv"), Path("fraiuddection.csv")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Dataset not found. Expected frauddetection.csv or fraiuddection.csv.")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    df[TARGET] = df[TARGET].map({"Y": 1, "N": 0}).astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ]
    )


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def main():
    data_path = resolve_dataset_path()
    df = load_data(data_path)

    X = df[FEATURES].copy()
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X)

    rf = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    candidates = {
        "RandomForest": rf,
        "XGBoost": xgb,
    }

    print("Training candidate models with requested 9 input fields...")
    test_results = []
    for name, model in candidates.items():
        metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        test_results.append(metrics)
        print(json.dumps(metrics, indent=2))

    test_results.sort(key=lambda x: (x["accuracy"], x["roc_auc"]), reverse=True)
    best_name = test_results[0]["model"]
    best_pipeline = candidates[best_name]
    print(f"\nSelected model: {best_name}")

    best_pipeline.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracy = cross_val_score(best_pipeline, X, y, scoring="accuracy", cv=cv, n_jobs=1)
    cv_auc = cross_val_score(best_pipeline, X, y, scoring="roc_auc", cv=cv, n_jobs=1)

    cv_summary = {
        "cv_accuracy_mean": float(np.mean(cv_accuracy)),
        "cv_accuracy_std": float(np.std(cv_accuracy)),
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
    }

    print("\nCross-validation summary:")
    print(json.dumps(cv_summary, indent=2))

    joblib.dump(best_pipeline, "fraud_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")

    with open("training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_features": FEATURES,
                "selected_model": best_name,
                "test_results": test_results,
                "cv_summary": cv_summary,
            },
            f,
            indent=2,
        )

    print("\nSaved fraud_model.pkl, preprocessor.pkl, and training_metrics.json")


if __name__ == "__main__":
    main()
