"""
train.py
--------
End-to-end training and evaluation for a single regressor.

Responsibilities of this module
--------------------------------
    1. Load the already-processed dataset produced by features.py
       (data/processed/ai4i2020_processed.csv). This module does not perform
       feature engineering; it only consumes the processed table.
    2. Validate the training table (target present, features numeric, no
       missing values) and separate X from y.
    3. Create a reproducible train/test split, fit a model, and evaluate with
       RMSE and R² on both the training and held-out test sets.
    4. Optionally persist the fitted model and a metrics JSON under
       models/saved/ so results survive past the current session. This is a
       local convenience only and is not a substitute for experiment tracking
       or a formal model registry.

Reusable entry-point
--------------------
train_baseline() accepts an optional `model` argument (default:
get_model("linear")). Comparison and tuning loops can therefore pass any
registered estimator — get_model("ridge"), get_model("random_forest"), … —
into the same function without duplicating load / split / fit / evaluate
logic.

random_state is forwarded automatically to any estimator that accepts it
(see _apply_random_state). Feature importance is extracted in a
model-agnostic way (see _extract_feature_importance) so both linear and
tree/ensemble models work with this function unchanged.

Run directly
------------
    python src/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Import sibling modules from src/ (same try/except pattern used in
# features.py, so every file under src/ resolves its neighbours consistently).
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
# Constants. PROCESSED_DIR and PROJECT_ROOT come from data_processing.py so
# the folder layout is defined in exactly one place across the project.
# ---------------------------------------------------------------------------
PROCESSED_PATH: Path = PROCESSED_DIR / "ai4i2020_processed.csv"
MODEL_DIR: Path = PROJECT_ROOT / "models" / "saved"
TARGET_COL: str = "rul_proxy_min"
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42

# R² gap (train − test) above which an overfitting note is printed.
# Not a hard rule — just a heads-up threshold for a human to look closer.
OVERFIT_R2_GAP_WARNING: float = 0.05


def load_processed(path: Path | str = PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed CSV produced by features.run_pipeline().

    Raises
    ------
    FileNotFoundError
        With a clear, actionable message if the processed file is missing.
    ValueError
        If the file exists but contains no rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run `python src/features.py` first to generate it."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Processed dataset is empty: {path}")
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the table and separate the feature matrix X from target y.

    Checks performed (fail loud, not silent)
    ----------------------------------------
    - Target column is present.
    - At least one feature column remains after dropping the target.
    - All feature columns are numeric.
    - No missing values in features or target.
    - Target is numeric and finite.

    Raises
    ------
    KeyError
        If the expected target column is missing.
    ValueError
        If the table is empty of features, contains missing values, or the
        target is non-finite.
    TypeError
        If any feature column or the target is non-numeric.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in processed data. "
            f"Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    if X.empty:
        raise ValueError("No feature columns remain after removing the target.")

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(
            "All training features must be numeric at this stage. "
            f"Non-numeric columns found: {non_numeric}"
        )

    if X.isna().any().any():
        missing_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Missing values found in feature columns: {missing_cols}. "
            "Handle them in the data/feature-processing stage before training."
        )

    inf_cols = X.columns[np.isinf(X).any()].tolist()
    if inf_cols:
        raise ValueError(f"Infinite values found in feature columns: {inf_cols}")

    if y.isna().any():
        raise ValueError("Missing values found in the regression target.")

    if not pd.api.types.is_numeric_dtype(y):
        raise TypeError(f"Target '{target_col}' must be numeric.")

    # NOTE: pandas.Series has no .isfinite() method (that's a numpy function,
    # not a Series method) — np.isfinite(y).all() is the correct form.
    if not np.isfinite(y).all():
        raise ValueError(f"Target '{target_col}' contains NaN or infinite values.")

    return X, y


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Compute RMSE and R².

    Returns a plain dict that is JSON-serialisable and ready for experiment
    tracking (e.g. mlflow.log_metrics(...)).
    """
    return {
        "rmse": float(_rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _apply_random_state(model: Any, random_state: int) -> Any:
    """Forward `random_state` to the estimator when it accepts one and the
    caller has not already set it explicitly.

    Without this, only the train/test split would be reproducible — a
    non-deterministic estimator (RandomForestRegressor, XGBRegressor, …)
    could still return different results run to run, and every call site
    would need to remember to pass random_state=... into get_model() itself.
    LinearRegression has no such parameter, so this is a no-op for the
    current baseline, but it keeps train_baseline() correct by default once
    other estimators are registered in models.py.
    """
    params = model.get_params()
    if "random_state" in params and params["random_state"] is None:
        model.set_params(random_state=random_state)
    return model


def _extract_feature_importance(
    model: Any, feature_names: list[str], X_train: pd.DataFrame
) -> dict[str, Any] | None:
    """Extract a model-agnostic feature-importance table, or None.

    Linear models (coef_)
    ----------------------
    Raw coefficients are in the original units of each feature, so they are
    directly interpretable ("+1 unit of wear_ratio changes the prediction by
    X minutes") but NOT comparable to each other for ranking — features here
    live on very different scales (type_encoded: 0–2 vs power_w: thousands of
    watts). To rank importance fairly, each coefficient is also scaled by
    that feature's train-set standard deviation (coef × std), which
    approximates the effect of a one-std change and is comparable across
    features. Both are returned: raw for interpretation, std-scaled for
    ranking.

    Tree / ensemble models (feature_importances_)
    -----------------------------------------------
    Already scale-invariant by construction, so used as-is. This branch is
    what makes train_baseline() work unchanged once get_model("random_forest")
    or similar is registered in models.py.

    Returns None for estimators exposing neither attribute.
    """
    if hasattr(model, "coef_"):
        feature_std = X_train[feature_names].std()
        raw_coef = pd.Series(model.coef_, index=feature_names)
        std_coef = raw_coef * feature_std
        table = pd.DataFrame({"raw_coef": raw_coef, "std_coef": std_coef})
        table = table.reindex(std_coef.abs().sort_values(ascending=False).index)
        return {
            "kind": "linear_coefficients",
            "table": table,
            "intercept": float(model.intercept_),
        }

    if hasattr(model, "feature_importances_"):
        table = pd.Series(model.feature_importances_, index=feature_names)
        table = table.sort_values(ascending=False).to_frame("importance")
        return {"kind": "feature_importances", "table": table, "intercept": None}

    return None


def train_baseline(
    model: Any = None,
    data_path: Path | str = PROCESSED_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """End-to-end training pipeline: load → validate → split → fit → evaluate → (save).

    Parameters
    ----------
    model : estimator, optional
        Any scikit-learn-compatible regressor (must implement .fit / .predict).
        Defaults to the registered baseline (LinearRegression). Comparison
        and tuning loops reuse this function by passing get_model("ridge"),
        get_model("random_forest"), etc. — no other change is required here.
    data_path : Path or str
        Path to the processed CSV (default: data/processed/ai4i2020_processed.csv).
    test_size : float
        Fraction of rows held out for testing (default 0.20). Must be in (0, 1).
    random_state : int
        Seed for the train/test split, and forwarded to the estimator itself
        when it accepts a random_state parameter that has not already been
        set (see _apply_random_state). Default 42.
    save : bool
        Whether to persist the fitted model and metrics JSON under
        models/saved/ (default True). Set False for quick experiments that
        should not be kept, or when looping over many models.
    verbose : bool
        Whether to print a step-by-step report.

    Returns
    -------
    dict
        {
          "model": the fitted estimator,
          "metrics": {"rmse": float, "r2": float},          # test set
          "train_metrics": {"rmse": float, "r2": float},    # train set,
              for spotting over/underfitting at a glance
          "feature_importance": dict | None,
              {"kind": ..., "table": pd.DataFrame, "intercept": float | None}
              — always populated when the estimator exposes coef_ or
              feature_importances_, regardless of verbose, so downstream code
              (plots, reports, experiment logging) can use it without scraping
              printed output,
          "feature_names": list[str],
          "n_train": int, "n_test": int,
          "X_train", "X_test", "y_train", "y_test", "preds":
              the split data and test-set predictions, so downstream steps
              (experiment logging, explainability, …) can reuse them without
              re-running the pipeline.
        }
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    if model is None:
        model = get_model(BASELINE_MODEL_NAME)

    if not hasattr(model, "fit") or not hasattr(model, "predict"):
        raise TypeError("model must implement both fit() and predict().")

    model = _apply_random_state(model, random_state)
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

    # 2. Validate + split
    if verbose:
        print("\n[2/5] Validating table and creating train/test split ...")
    X, y = split_features_target(df)
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if verbose:
        print(f"      Features : {feature_names}")
        print(f"      Train    : {len(X_train):,} rows")
        print(f"      Test     : {len(X_test):,} rows")
        print(f"      Test size: {test_size:.0%}")
        print(f"      Seed     : {random_state}")

    # 3. Fit
    if verbose:
        print("\n[3/5] Fitting model ...")
    model.fit(X_train, y_train)
    if verbose:
        print("      Done.")

    # 4. Evaluate on both splits so over/underfitting is visible at a glance
    # rather than only ever seeing the test-set number in isolation.
    if verbose:
        print("\n[4/5] Evaluating on train and held-out test sets ...")
    train_preds = model.predict(X_train)
    preds = model.predict(X_test)
    train_metrics = evaluate_regression(y_train, train_preds)
    metrics = evaluate_regression(y_test, preds)
    if verbose:
        print(
            f"      Train  RMSE : {train_metrics['rmse']:.4f}   "
            f"R² : {train_metrics['r2']:.4f}"
        )
        print(
            f"      Test   RMSE : {metrics['rmse']:.4f}   "
            f"R² : {metrics['r2']:.4f}"
        )
        r2_gap = train_metrics["r2"] - metrics["r2"]
        if r2_gap > OVERFIT_R2_GAP_WARNING:
            print(
                f"      Note: train R² exceeds test R² by {r2_gap:.3f} — "
                "worth watching for overfitting as model complexity increases."
            )

    # Model-agnostic feature importance. Always computed (not only when
    # verbose) so callers get it back in the return dict either way.
    importance = _extract_feature_importance(model, feature_names, X_train)
    if importance is not None and verbose:
        if importance["kind"] == "linear_coefficients":
            print(
                "\n      Coefficients "
                "(sorted by |standardized coef| — comparable across feature scales):"
            )
            for name, row in importance["table"].iterrows():
                print(
                    f"        {name:30s}  "
                    f"raw={row['raw_coef']:+.6f}   std={row['std_coef']:+.6f}"
                )
            print(
                f"        {'intercept':30s}  "
                f"raw={importance['intercept']:+.6f}"
            )
        else:
            print("\n      Feature importances (sorted):")
            for name, val in importance["table"]["importance"].items():
                print(f"        {name:30s}  {val:.6f}")

    # 5. Persist (local convenience only; formal model registration belongs
    # to the experiment-tracking layer).
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
                "train_metrics": train_metrics,
                "random_state": random_state,
                "test_size": test_size,
            },
            model_path,
        )
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model": model_name,
                    "test_metrics": metrics,
                    "train_metrics": train_metrics,
                    "n_samples": len(df),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_features": len(feature_names),
                    "feature_names": feature_names,
                    "target_col": TARGET_COL,
                    "random_state": random_state,
                    "test_size": test_size,
                    "feature_importance": (
                        importance["table"].round(6).to_dict(orient="index")
                        if importance is not None
                        else None
                    ),
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
        "train_metrics": train_metrics,
        "feature_importance": importance,
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
