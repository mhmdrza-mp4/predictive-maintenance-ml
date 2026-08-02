"""
train.py
--------
Baseline model training: Linear Regression, end-to-end.

Responsibility of this module (single responsibility, per project architecture):
    1. Load the already-processed dataset (data/processed/ai4i2020_processed.csv)
       produced by features.py. This module does NOT redo feature engineering;
       it only consumes the processed table.
    2. Split into train/test, fit a model, and evaluate with RMSE and R².
    3. Persist the fitted model + a small metrics JSON to models/saved/, as a
       convenience so the baseline survives past the current session. This is
       not a replacement for MLflow's Model Registry — it simply means the
       baseline is not lost when the terminal closes.

Design note — reusable training entry-point:
    train_baseline() accepts an optional `model` argument (default:
    get_model("linear")) so that algorithm-comparison loops can pass any
    registered estimator (get_model("ridge"), get_model("random_forest"), …)
    into this same function without duplicating the load/split/fit/evaluate
    logic.

Run directly:
    python src/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Import sibling modules from src/ (same try/except pattern already used in
# features.py, so every file in src/ resolves its neighbours the same way).
# ---------------------------------------------------------------------------
try:
    from data_processing import PROCESSED_DIR, PROJECT_ROOT
    from models import BASELINE_MODEL_NAME, get_model
except ImportError:
    _src_dir = str(Path(__file__).resolve().parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from data_processing import PROCESSED_DIR, PROJECT_ROOT
    from models import BASELINE_MODEL_NAME, get_model

# ---------------------------------------------------------------------------
# RMSE helper — scikit-learn >= 1.4 prefers root_mean_squared_error and has
# dropped `squared=False` from mean_squared_error in some releases. Falling
# back to a manual sqrt (rather than relying on the `squared` kwarg still
# existing) keeps this working across the widest possible range of versions.
# ---------------------------------------------------------------------------
try:
    from sklearn.metrics import root_mean_squared_error as _rmse
except ImportError:  # scikit-learn < 1.4
    from sklearn.metrics import mean_squared_error as _mse

    def _rmse(y_true, y_pred) -> float:
        return float(_mse(y_true, y_pred) ** 0.5)


# ---------------------------------------------------------------------------
# Constants. PROCESSED_DIR / PROJECT_ROOT come from data_processing.py so the
# folder layout is defined in exactly one place in the whole project.
# ---------------------------------------------------------------------------
PROCESSED_PATH: Path = PROCESSED_DIR / "ai4i2020_processed.csv"
MODEL_DIR: Path = PROJECT_ROOT / "models" / "saved"
TARGET_COL: str = "rul_proxy_min"
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42


def load_processed(path: Path | str = PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed CSV produced by features.run_pipeline().

    Raises
    ------
    FileNotFoundError
        With a clear, actionable message if the processed file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run `python src/features.py` first to generate it."
        )
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the target vector y.

    Raises
    ------
    KeyError
        If the expected target column is missing (fail loud, not silent).
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in processed data. "
            f"Available columns: {df.columns.tolist()}"
        )
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Compute RMSE and R².

    Returns a plain dict that is JSON-serialisable and MLflow-ready
    (`mlflow.log_metrics(evaluate_regression(y_test, preds))` works as-is).
    """
    return {"rmse": float(_rmse(y_true, y_pred)), "r2": float(r2_score(y_true, y_pred))}


def train_baseline(
    model: Any = None,
    data_path: Path | str = PROCESSED_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end baseline pipeline: load → split → train → evaluate → (save).

    Parameters
    ----------
    model : estimator, optional
        Any scikit-learn-compatible regressor (must implement .fit / .predict).
        Defaults to the registered baseline (LinearRegression). Comparison
        loops reuse this function by passing get_model("ridge"),
        get_model("random_forest"), etc. — no other change needed here.
    data_path : Path or str
        Path to the processed CSV (default: data/processed/ai4i2020_processed.csv).
    test_size : float
        Fraction of rows held out for testing (default 0.20).
    random_state : int
        Seed for the train/test split, for reproducible results (default 42).
    save : bool
        Whether to persist the fitted model + metrics JSON under
        models/saved/ (default True). Set False for quick experiments you
        do not want to keep, or when looping over many models.
    verbose : bool
        Whether to print a step-by-step report.

    Returns
    -------
    dict
        {
          "model": the fitted estimator,
          "metrics": {"rmse": float, "r2": float},
          "feature_names": list[str],
          "n_train": int, "n_test": int,
          "X_train", "X_test", "y_train", "y_test", "preds": the split data
          and predictions, so downstream steps (MLflow logging, SHAP, …)
          can reuse them without re-running the pipeline.
        }
    """
    if model is None:
        model = get_model(BASELINE_MODEL_NAME)
    model_name = type(model).__name__

    if verbose:
        print("=" * 60)
        print(f"Baseline training ({model_name})")
        print("=" * 60)

    # 1. Load
    if verbose:
        print("\n[1/5] Loading processed data ...")
    df = load_processed(data_path)
    if verbose:
        print(f"      Shape : {df.shape}")

    # 2. Split
    if verbose:
        print("\n[2/5] Splitting features/target and train/test ...")
    X, y = split_features_target(df)
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if verbose:
        print(f"      Features : {feature_names}")
        print(f"      Train    : {len(X_train):,} rows")
        print(f"      Test     : {len(X_test):,} rows")

    # 3. Fit
    if verbose:
        print("\n[3/5] Fitting model ...")
    model.fit(X_train, y_train)
    if verbose:
        print("      Done.")

    # 4. Evaluate
    if verbose:
        print("\n[4/5] Evaluating on held-out test set ...")
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)
    if verbose:
        print(f"      RMSE : {metrics['rmse']:.4f}  (minutes — lower is better)")
        print(f"      R²   : {metrics['r2']:.4f}    (fraction of variance explained)")

    # Coefficient summary — linear models only. Useful for interpreting the
    # proxy-target construction noted in features.py: wear_ratio / overstrain
    # are expected to dominate since rul_proxy_min is derived from tool wear.
    if hasattr(model, "coef_") and verbose:
        coef_table = pd.Series(model.coef_, index=feature_names).sort_values(
            key=abs, ascending=False
        )
        print("\n      Coefficients (sorted by |value|):")
        for name, val in coef_table.items():
            print(f"        {name:30s}  {val:+.6f}")
        print(f"        {'intercept':30s}  {model.intercept_:+.6f}")

    # 5. Persist (local convenience only; the official "final model" lives in
    # the MLflow Model Registry once experiment tracking is wired up).
    if save:
        if verbose:
            print("\n[5/5] Saving model and metrics ...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.lower()
        model_path = MODEL_DIR / f"{safe_name}_baseline.joblib"
        metrics_path = MODEL_DIR / f"{safe_name}_baseline_metrics.json"

        joblib.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "target_col": TARGET_COL,
                "metrics": metrics,
                "random_state": random_state,
                "test_size": test_size,
            },
            model_path,
        )
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model": model_name,
                    "metrics": metrics,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "feature_names": feature_names,
                    "random_state": random_state,
                    "test_size": test_size,
                    "notes": (
                        "rul_proxy_min is a deterministic transform of Tool wear. "
                        "wear_ratio and overstrain still carry tool-wear signal, "
                        "so the high R² reflects that construction and is not "
                        "evidence of an independently learned physical pattern."
                    ),
                },
                fh,
                indent=2,
            )
        if verbose:
            print(f"      Model   → {model_path}")
            print(f"      Metrics → {metrics_path}")
    elif verbose:
        print("\n[5/5] Skipping save (save=False).")

    if verbose:
        print("\nDone.")

    return {
        "model": model,
        "metrics": metrics,
        "feature_names": feature_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preds": preds,
    }


if __name__ == "__main__":
    try:
        train_baseline()
    except Exception as exc:  # non-zero exit on failure keeps CI / terminal honest
        print(f"\nBaseline training FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
