"""
models.py
---------
Model registry and factory.

Responsibility of this module (single responsibility, per project architecture):
    Construct fresh, unfitted scikit-learn-compatible regressors by name, so
    that train.py, stacking, and hyperparameter-tuning code never hard-code
    estimator setup in more than one place.

Why the registry stores classes (not lambdas or instances):
    Cross-validation and hyperparameter search both need a *fresh, unfitted*
    estimator for every fold / trial. Storing the class itself and calling it
    with **kwargs guarantees a brand-new instance each time get_model() is
    invoked, AND lets a caller forward constructor arguments straight through
    — e.g. get_model("ridge", alpha=0.5) — which an Optuna (or GridSearch)
    loop needs in order to try different hyperparameter values without ever
    touching this file again.

Extending the registry:
    Adding a new algorithm is a one-line change in _MODEL_REGISTRY (import the
    class at the top, add its entry below). No other file in the project needs
    to change.

Naming convention for registry keys: short, lowercase identifiers
("linear", "ridge", "lasso", "svr", "random_forest", "xgboost") so callers
and the registry stay aligned 1:1.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Registry: short name -> estimator class (not an instance!)
# ---------------------------------------------------------------------------
# Currently registered: LinearRegression (the project baseline).
# Additional estimators (Ridge, Lasso, SVR, RandomForest, XGBoost, …) are
# added here as the modelling pipeline expands — import each class above and
# register one line below.
_MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "linear": LinearRegression,
    # "ridge":         Ridge,
    # "lasso":         Lasso,
    # "svr":           SVR,
    # "random_forest": RandomForestRegressor,
    # "xgboost":       XGBRegressor,
}

# Canonical baseline model for this project.
BASELINE_MODEL_NAME: str = "linear"


def get_model(name: str = BASELINE_MODEL_NAME, **kwargs: Any) -> BaseEstimator:
    """Return a fresh, unfitted regressor by short name.

    Parameters
    ----------
    name : str
        A key in _MODEL_REGISTRY (case-insensitive). Defaults to the project
        baseline ("linear").
    **kwargs
        Forwarded to the underlying constructor, e.g.
        get_model("ridge", alpha=0.5) inside a tuning trial, or
        get_model("linear", fit_intercept=False) to try a variant.

    Returns
    -------
    sklearn.base.BaseEstimator
        A new, unfitted estimator ready for .fit(X, y).

    Raises
    ------
    ValueError
        If `name` is not registered. Fails loud with the list of valid names
        instead of returning None or silently picking a default.
    """
    key = name.strip().lower()
    if key not in _MODEL_REGISTRY:
        known = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Known models: {known}")
    return _MODEL_REGISTRY[key](**kwargs)


def list_models() -> list[str]:
    """Return the sorted list of currently registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


if __name__ == "__main__":
    print("Registered models:")
    for m in list_models():
        print(f"  {m!r:20s} -> {get_model(m).__class__.__name__}")
