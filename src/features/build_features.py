#!/usr/bin/env python3
"""
Production-like training pipeline using scikit-learn pipelines,
hyperparameter search, strong logging, and model artifact saving.

Usage:
    python train.py --train data/raw/train.csv --test data/raw/test.csv \
        --output-dir models/ --target-col target --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)
from scipy.stats import randint, uniform

# Keep your own `feature_build` import (unchanged)
from feature_definitions import feature_build

# Basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("ml.training")

# ---------- Config dataclass ----------
@dataclass
class TrainConfig:
    train_path: str
    test_path: str
    output_dir: str
    target_col: str = "target"
    random_seed: int = 42
    test_size: float = 0.2
    n_iter_search: int = 25
    cv_folds: int = 5
    refit: bool = True
    n_jobs: int = -1
    verbose: int = 1


# ---------- Utility helpers ----------
def load_data(path: str) -> pd.DataFrame:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    LOGGER.info("Loading data from %s", path)
    return pd.read_csv(path)


def detect_task(y: pd.Series) -> str:
    """
    Decide whether the problem is classification or regression.
    Heuristic: if dtype is integer or object (small unique values) -> classification.
    """
    if pd.api.types.is_float_dtype(y):
        return "regression"
    if pd.api.types.is_integer_dtype(y):
        return "classification"
    # If object or category or small unique counts -> classification
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"
    # fallback
    unique_ratio = y.nunique() / len(y)
    if unique_ratio < 0.05:
        return "classification"
    return "regression"


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """
    Build a ColumnTransformer for numeric and categorical features.

    Returns:
        preprocessor: ColumnTransformer instance
        numeric_features: List[str]
        categorical_features: List[str]
    """
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    LOGGER.info("Detected numeric features: %s", numeric_features)
    LOGGER.info("Detected categorical features: %s", categorical_features)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # drop other columns
        sparse_threshold=0.0,
    )
    return preprocessor, numeric_features, categorical_features


def build_estimator(task: str, random_state: int = 42):
    """
    Create a default estimator for classification or regression.
    You can replace with XGBoost/CatBoost/LightGBM if installed.
    """
    if task == "classification":
        # For speed + great default behaviour we use HistGradientBoostingClassifier (handles missing natively),
        # but to remain compatible with scikit-learn pipelines and column transform we use RandomForest as baseline.
        est = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        est = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    return est


def build_pipeline(X: pd.DataFrame, task: str, random_state: int = 42) -> Pipeline:
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    estimator = build_estimator(task, random_state=random_state)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return pipeline


# ---------- Training/evaluation ----------
def train_and_tune(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    cfg: TrainConfig,
) -> Tuple[Pipeline, Optional[RandomizedSearchCV]]:
    """
    Train pipeline and perform randomized hyperparameter search.

    Returns:
        best_pipeline: fitted pipeline (if RandomizedSearchCV used, returns best_estimator_)
        search_obj: the RandomizedSearchCV object or None
    """
    LOGGER.info("Starting cross-validated hyperparameter search (n_iter=%d)", cfg.n_iter_search)

    # Default parameter grid for RandomForest. Extend for other models.
    if task == "classification":
        param_distributions = {
            "estimator__n_estimators": randint(100, 500),
            "estimator__max_depth": randint(3, 50),
            "estimator__min_samples_split": randint(2, 10),
            "estimator__min_samples_leaf": randint(1, 10),
            "estimator__max_features": ["sqrt", "log2", 0.5, 0.8],
        }
    else:
        param_distributions = {
            "estimator__n_estimators": randint(100, 500),
            "estimator__max_depth": randint(3, 50),
            "estimator__min_samples_split": randint(2, 10),
            "estimator__min_samples_leaf": randint(1, 10),
            "estimator__max_features": ["sqrt", "log2", 0.5, 0.8],
        }

    cv = cfg.cv_folds
    # For classification, use stratified k-fold
    if task == "classification":
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=cfg.random_seed)
    else:
        # regression fallback to simple KFold via cross_val_score when needed
        skf = cv

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=cfg.n_iter_search,
        cv=skf,
        verbose=cfg.verbose,
        random_state=cfg.random_seed,
        n_jobs=cfg.n_jobs,
        return_train_score=False,
        refit=cfg.refit,
    )
    search.fit(X_train, y_train)
    LOGGER.info("Best params: %s", search.best_params_)
    LOGGER.info("Best CV score: %f", search.best_score_)
    # Return the best pipeline
    best_pipeline = search.best_estimator_
    return best_pipeline, search


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y_true: pd.Series, task: str) -> Dict[str, float]:
    """
    Evaluate the pipeline on a holdout/test set and return metrics dict.
    """
    LOGGER.info("Evaluating on holdout set (n=%d)", len(y_true))
    y_pred = pipeline.predict(X)
    metrics: Dict[str, float] = {}
    if task == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        try:
            if len(np.unique(y_true)) == 2:
                y_proba = pipeline.predict_proba(X)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception as e:
            LOGGER.warning("ROC AUC could not be computed: %s", e)
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"] = float(r2_score(y_true, y_pred))
    LOGGER.info("Evaluation metrics: %s", metrics)
    return metrics


def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, float],
    output_dir: str,
    model_name: str = "model.joblib",
) -> Tuple[str, str]:
    """
    Save the trained pipeline and metrics to output_dir.
    Returns (model_path, metrics_path)
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / model_name
    metrics_path = output_dir / "metrics.json"

    LOGGER.info("Saving model to %s", model_path)
    joblib.dump(pipeline, model_path)

    LOGGER.info("Saving metrics to %s", metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return str(model_path), str(metrics_path)


def run_training(cfg: TrainConfig) -> None:
    # 1) Load raw data
    train_raw = load_data(cfg.train_path)
    test_raw = load_data(cfg.test_path)

    # 2) Feature building (your implementation)
    LOGGER.info("Building features for training set")
    train_features = feature_build(train_raw, mode="train")
    LOGGER.info("Building features for test set")
    test_features = feature_build(test_raw, mode="test")

    # 3) Ensure target exists
    if cfg.target_col not in train_features.columns:
        raise KeyError(f"Target column '{cfg.target_col}' not found in training data after feature_build.")
    if cfg.target_col not in test_features.columns:
        LOGGER.warning("Target column '%s' not found in test data (test may not include labels).", cfg.target_col)

    X = train_features.drop(columns=[cfg.target_col])
    y = train_features[cfg.target_col]

    # 4) Detect task and split train/val
    task = detect_task(y)
    LOGGER.info("Detected task: %s", task)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y if task == "classification" else None
    )

    # 5) build pipeline
    pipeline = build_pipeline(X_train, task=task, random_state=cfg.random_seed)

    # 6) Train + tune
    best_pipeline, search_obj = train_and_tune(pipeline, X_train, y_train, task, cfg)

    # 7) Evaluate on val & test (if labels present)
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, task)

    # Evaluate on test set only if target exists
    metrics = {"validation": val_metrics}
    if cfg.target_col in test_features.columns:
        X_test = test_features.drop(columns=[cfg.target_col])
        y_test = test_features[cfg.target_col]
        test_metrics = evaluate_model(best_pipeline, X_test, y_test, task)
        metrics["test"] = test_metrics
    else:
        LOGGER.info("No labels for test set; skipping test evaluation.")

    # 8) Save model & metrics
    model_path, metrics_path = save_artifacts(best_pipeline, metrics, cfg.output_dir)
    LOGGER.info("Saved model to %s and metrics to %s", model_path, metrics_path)


# ---------- Prediction wrapper ----------
def predict_from_saved_model(model_path: str, df: pd.DataFrame) -> np.ndarray:
    """
    Load a saved pipeline/model and predict on df (raw features expected).
    """
    if not pathlib.Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline: Pipeline = joblib.load(model_path)
    return pipeline.predict(df)


# ---------- CLI ----------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and save an sklearn model pipeline.")
    parser.add_argument("--train", required=True, help="Path to raw train CSV")
    parser.add_argument("--test", required=True, help="Path to raw test CSV")
    parser.add_argument("--target-col", default="target", help="Name of target column after feature_build")
    parser.add_argument("--output-dir", default="models", help="Directory to save trained model and metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-iter", type=int, default=25, help="Number of iterations for RandomizedSearchCV")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    return parser


def main(argv=sys.argv[1:]) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        target_col=args.target_col,
        random_seed=args.seed,
        n_iter_search=args.n_iter,
        cv_folds=args.cv_folds,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
