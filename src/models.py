"""
models.py
---------
Model registry and factory.

Responsibilities of this module
--------------------------------
    Construct fresh, unfitted scikit-learn-compatible regressors by short
    name, so that training, comparison, stacking, and hyperparameter-search
    code never hard-code estimator construction in more than one place.

Why the registry stores classes, not instances or lambdas
---------------------------------------------------------
Cross-validation and hyperparameter search both require a *fresh, unfitted*
estimator for every fold or trial. Storing the class itself and instantiating
it with **kwargs guarantees a brand-new object on each call to get_model(),
and lets a caller forward constructor arguments directly — e.g.
get_model("ridge", alpha=0.5) — without this module knowing about the search
loop.

Extending the registry
----------------------
Adding an algorithm is a one-line change: import the class at the top of
this file and register it in _MODEL_REGISTRY. No other module needs to change.

Naming convention
-----------------
Registry keys are short, lowercase identifiers
("linear", "ridge", "lasso", "svr", "random_forest", "xgboost") so that
callers and the registry stay aligned one-to-one.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Registry: short name -> estimator class (never an instance)
# ---------------------------------------------------------------------------
# Only LinearRegression is registered at present (the project baseline).
# Additional estimators are added here as they are introduced into the
# modelling pipeline — import the class above, then register one line below.
_MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "linear": LinearRegression,
    # "ridge":         Ridge,
    # "lasso":         Lasso,
    # "svr":           SVR,
    # "random_forest": RandomForestRegressor,
    # "xgboost":       XGBRegressor,
}

# Default model used when no name is supplied to get_model().
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
