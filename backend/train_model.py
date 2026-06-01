from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "DEATH_EVENT"
TIME_COLUMN = "time"
DEFAULT_DATA_FILES = (
    "heart_failure_prediction.csv",
    "heart_failure_clinical_records_dataset.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save the heart failure mortality prediction model."
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-path", default="backend/model/heart_failure_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--tuning-iterations", type=int, default=8)
    return parser.parse_args()


def resolve_data_path(data_path: str | None) -> Path:
    if data_path:
        return Path(data_path)

    for candidate in DEFAULT_DATA_FILES:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    return Path(DEFAULT_DATA_FILES[0])


def make_scaled_classifier(classifier: Any) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )


def make_tree_classifier(classifier: Any) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )


def build_base_models(y_train: pd.Series) -> dict[str, ImbPipeline]:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    return {
        "Logistic Regression": make_scaled_classifier(
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": make_tree_classifier(
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "XGBoost": make_tree_classifier(
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "SVM": make_scaled_classifier(
            SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
            )
        ),
    }


def tune_base_models(
    base_models: dict[str, ImbPipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_iterations: int,
) -> tuple[dict[str, ImbPipeline], pd.DataFrame]:
    tuning_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    tuning_scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    search_spaces = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "classifier__penalty": ["l1", "l2"],
        },
        "Random Forest": {
            "classifier__n_estimators": [300, 500, 800],
            "classifier__max_depth": [None, 3, 5, 8],
            "classifier__min_samples_leaf": [1, 2, 4, 6],
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            "classifier__reg_lambda": [1, 2, 5],
        },
        "SVM": {
            "classifier__C": [0.1, 0.3, 1, 3, 10],
            "classifier__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
        },
    }

    tuned_models: dict[str, ImbPipeline] = {}
    rows = []

    for model_name, model in base_models.items():
        print(f"Tuning {model_name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=search_spaces[model_name],
            n_iter=tuning_iterations,
            scoring=tuning_scoring,
            refit="f1",
            cv=tuning_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        best_index = search.best_index_
        tuned_models[model_name] = search.best_estimator_
        rows.append(
            {
                "base_model": model_name,
                "best_cv_f1": float(search.best_score_),
                "best_cv_recall": float(search.cv_results_["mean_test_recall"][best_index]),
                "best_cv_precision": float(
                    search.cv_results_["mean_test_precision"][best_index]
                ),
                "best_cv_roc_auc": float(
                    search.cv_results_["mean_test_roc_auc"][best_index]
                ),
                "best_parameters": search.best_params_,
            }
        )

    tuning_results = pd.DataFrame(rows).sort_values(
        by=["best_cv_f1", "best_cv_recall", "best_cv_precision"], ascending=False
    )
    return tuned_models, tuning_results


def build_stacking_model(tuned_base_models: dict[str, ImbPipeline]) -> StackingClassifier:
    stacking_estimators = [
        ("lr", tuned_base_models["Logistic Regression"]),
        ("rf", tuned_base_models["Random Forest"]),
        ("xgb", tuned_base_models["XGBoost"]),
        ("svm", tuned_base_models["SVM"]),
    ]

    return StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
        passthrough=True,
    )


def choose_threshold(
    model: StackingClassifier, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    stack_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stack_scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=stack_scoring,
        cv=stack_cv,
        n_jobs=-1,
        error_score="raise",
    )
    cv_results = pd.DataFrame(
        [
            {
                "metric": metric_name,
                "mean": float(np.mean(cv_scores[f"test_{metric_name}"])),
                "std": float(np.std(cv_scores[f"test_{metric_name}"])),
            }
            for metric_name in stack_scoring
        ]
    )

    out_of_fold_probability = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=stack_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        threshold_prediction = (out_of_fold_probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(y_train, threshold_prediction, zero_division=0)
                ),
                "recall": float(recall_score(y_train, threshold_prediction, zero_division=0)),
                "f1": float(f1_score(y_train, threshold_prediction, zero_division=0)),
            }
        )

    threshold_results = pd.DataFrame(rows).sort_values(
        by=["f1", "recall", "precision"], ascending=False
    )
    threshold = float(threshold_results.iloc[0]["threshold"])
    return threshold, threshold_results, cv_results


def evaluate(
    model: StackingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, float | int | str]:
    y_probability = model.predict_proba(X_test)[:, 1]
    y_predicted = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

    return {
        "model": "SMOTE Tuned Stacking Ensemble",
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_predicted)),
        "precision": float(precision_score(y_test, y_predicted, zero_division=0)),
        "recall": float(recall_score(y_test, y_predicted, zero_division=0)),
        "f1": float(f1_score(y_test, y_predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
        "average_precision": float(average_precision_score(y_test, y_probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    drop_columns = [TARGET_COLUMN]
    if not args.include_time:
        drop_columns.append(TIME_COLUMN)

    X = df.drop(columns=drop_columns)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    base_models = build_base_models(y_train)
    tuned_base_models, tuning_results = tune_base_models(
        base_models, X_train, y_train, args.tuning_iterations
    )
    model = build_stacking_model(tuned_base_models)
    threshold, threshold_results, cv_results = choose_threshold(model, X_train, y_train)

    print(f"Optimized threshold: {threshold:.2f}")
    model.fit(X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test, threshold)

    artifact = {
        "model": model,
        "threshold": threshold,
        "features": list(X.columns),
        "target": TARGET_COLUMN,
        "include_time": args.include_time,
        "test_size": args.test_size,
        "random_state": RANDOM_STATE,
        "test_metrics": test_metrics,
        "cv_results": cv_results.to_dict(orient="records"),
        "threshold_results": threshold_results.head(10).to_dict(orient="records"),
        "tuning_results": tuning_results.to_dict(orient="records"),
    }

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    print(f"Saved model artifact to {model_path}")
    print(pd.DataFrame([test_metrics]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
