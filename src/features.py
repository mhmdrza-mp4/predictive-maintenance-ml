"""
features.py
-----------
Phase 2 (PH.02) — Feature engineering and proxy regression-target construction.

MOST IMPORTANT DECISION IN THIS FILE: the regression target.
    AI4I 2020 is natively a classification dataset (binary "Machine failure").
    There is no column for "time remaining until failure" (no true RUL label).
    Per the project roadmap (Path 1 — same dataset, documented as a proxy):

        rul_proxy_min = max(0, 200 - Tool wear [min])

    200 min is the conservative (earliest) edge of the documented TWF risk
    window [200, 240] (DatasetDoc.txt: "the tool will be replaced or fail at
    a randomly selected tool wear time between 200-240 mins").

    This is an HONEST PROXY, not a ground-truth Remaining Useful Life label:
    - It is a deterministic, monotonic transform of a single existing raw
      column, not an independently measured outcome.
    - It only captures tool-wear degradation (TWF), not the other four
      failure modes (HDF, PWF, OSF, RNF).
    - If a "true" degradation-time regression target is required later, the
      roadmap's Path 2 (NASA CMAPSS, which has a real RUL column) is the
      correct dataset — not this one.
    This caveat must be restated in the README and Model Card, per the roadmap.

    Consequence for feature selection: because rul_proxy_min is a deterministic,
    monotonic function of "Tool wear [min]" alone, that raw column is dropped
    from the modelling features (see LEAKAGE_COLS below) — keeping it would let
    any model "solve" the task by learning the identity y = 200 - x instead of
    an actual pattern, making every downstream comparison (PH.03 baseline,
    PH.06 tuning, PH.07 stacking) meaningless. wear_ratio and overstrain are
    kept, since the roadmap asks for them explicitly and they are legitimate
    physics-informed features in their own right, but note that both remain
    strongly correlated with the target by construction (~-0.98 and ~-0.90
    respectively, since both are built from tool wear too). This should be
    flagged in the README/model card, and feature-importance results in PH.08
    should be read with this in mind rather than as evidence of a "learned"
    pattern.

NOT implemented here: rolling windows / moving averages.
    AI4I 2020 has no time-series structure to roll over: each row is an
    independent snapshot (Product ID is unique per row, UDI is a row index,
    not a timestamp — confirmed in DatasetDoc.txt and the roadmap's own
    domain notes, which is also why TimeSeriesSplit was deliberately rejected
    for this project). A "moving average" would silently average across
    unrelated machines, which is a technically wrong claim to ship. The
    ratio/interaction features below are used instead, in line with the
    roadmap's own examples (temp_diff, power, wear_ratio).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import data_processing from the same package (src/)
# ---------------------------------------------------------------------------
# When running as a script from project root, src/ may need to be on path.
# When imported as a module, this works naturally.
# ---------------------------------------------------------------------------
try:
    from data_processing import (
        PROJECT_ROOT,
        PROCESSED_DIR,
        load_and_validate,
    )
except ImportError:
    # If imported from project root where src/ is not in PYTHONPATH
    _src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from data_processing import (
        PROJECT_ROOT,
        PROCESSED_DIR,
        load_and_validate,
    )

# ---------------------------------------------------------------------------
# Constants derived directly from DatasetDoc.txt (not arbitrary guesses)
# ---------------------------------------------------------------------------

# TWF risk window start: "a randomly selected tool wear time between 200-240 mins"
# 200 is used as the conservative (earliest) entry point into the risk window.
TWF_RISK_WINDOW_START_MIN: int = 200

# OSF thresholds are variant-specific: tool_wear * torque exceeds this value.
OSF_LIMITS: dict[str, float] = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}

# Output path for the processed dataset
PROCESSED_PATH: Path = PROCESSED_DIR / "ai4i2020_processed.csv"

# Columns that must be dropped before modelling to prevent label leakage or
# because they are pure identifiers / already encoded.
LEAKAGE_COLS: list[str] = [
    "UDI",
    "Product ID",
    "Type",               # encoded as type_encoded
    "Machine failure",    # outcome label — using it would be leakage
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Tool wear [min]",     # rul_proxy_min = max(0, 200 - this column), so
                            # keeping it would let a model trivially reconstruct
                            # the target instead of learning a real pattern
                            # (measured correlation with rul_proxy_min: -0.998)
]


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """Encode product quality variant L/M/H as an ordinal integer.

    Mapping: L → 0, M → 1, H → 2.
    This ordering is physically meaningful: the dataset documentation states
    that L, M, and H add 2, 3, and 5 minutes of tool wear per cycle,
    respectively. An ordinal encoding preserves this monotonic relationship
    in a single compact column.
    """
    df = df.copy()
    mapping = {"L": 0, "M": 1, "H": 2}
    df["type_encoded"] = df["Type"].map(mapping).astype(int)
    return df


def add_temp_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Process minus air temperature [K].

    This difference is the exact quantity that drives HDF (Heat Dissipation
    Failure): HDF triggers when temp_diff < 8.6 K AND rotational speed
    < 1380 rpm (per DatasetDoc.txt).
    """
    df = df.copy()
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    return df


def add_power(df: pd.DataFrame) -> pd.DataFrame:
    """Mechanical power in Watts: torque [Nm] × angular speed [rad/s].

    Angular speed ω = rpm × 2π / 60.
    PWF (Power Failure) triggers when power < 3500 W or > 9000 W.
    """
    df = df.copy()
    omega_rad_s = df["Rotational speed [rpm]"] * (2.0 * np.pi / 60.0)
    df["power_w"] = df["Torque [Nm]"] * omega_rad_s
    return df


def add_overstrain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Overstrain interaction and safety margin.

    overstrain = tool_wear [min] × torque [Nm]
    OSF triggers when overstrain exceeds the variant-specific limit
    (11 k / 12 k / 13 k min·Nm for L / M / H).

    osf_margin = limit - overstrain
        positive  → still safe
        zero/negative → at or past OSF threshold
    """
    df = df.copy()
    df["overstrain"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    osf_limit = df["Type"].map(OSF_LIMITS)
    df["osf_margin"] = osf_limit - df["overstrain"]
    return df


def add_wear_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Tool wear intensity relative to rotational speed.

    wear_ratio = tool_wear [min] / rotational_speed [rpm]

    Rotational speed is guaranteed > 0 by the Pandera raw schema, so division
    by zero is impossible for validated data.
    """
    df = df.copy()
    df["wear_ratio"] = df["Tool wear [min]"] / df["Rotational speed [rpm]"]
    return df


def build_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    """Build the proxy regression target.

    rul_proxy_min = max(0, 200 - Tool wear [min])

    Interpretation
    --------------
    • 200  → brand-new tool (0 min wear).
    • 0    → tool has reached or exceeded the 200-minute service threshold.
    • Between → estimated minutes of useful life left before planned service.
    """
    df = df.copy()
    df["rul_proxy_min"] = np.maximum(
        0, TWF_RISK_WINDOW_START_MIN - df["Tool wear [min]"]
    ).astype(int)
    return df


def drop_leakage_and_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier, raw-categorical, and outcome-leakage columns."""
    return df.drop(columns=LEAKAGE_COLS, errors="ignore")


def sanity_check_processed(df: pd.DataFrame) -> None:
    """Basic post-processing sanity checks.

    NOT a Pandera schema — the schema guards the *raw* boundary;
    these checks guard the *processed* boundary.
    """
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Unexpected NaN values found in processed data: {nan_cols}")

    target = df["rul_proxy_min"]
    if target.min() < 0:
        raise ValueError("Proxy target contains negative values — clipping logic failed.")
    if target.max() > TWF_RISK_WINDOW_START_MIN:
        raise ValueError(
            f"Proxy target exceeds {TWF_RISK_WINDOW_START_MIN} — "
            "this should be impossible given the clipping logic."
        )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Validated RAW dataframe from data_processing.load_and_validate().

    Returns
    -------
    pd.DataFrame
        Processed dataframe ready for modelling (features + proxy target,
        no leakage columns).
    """
    # Defensive: work on a copy so the input is never mutated
    df = df.copy()

    # 1. Encode categorical
    df = encode_type(df)

    # 2. Physics-informed interaction features
    df = add_temp_diff(df)
    df = add_power(df)
    df = add_overstrain_features(df)
    df = add_wear_ratio(df)

    # 3. Construct proxy target
    df = build_regression_target(df)

    # 4. Drop leakage / redundant columns
    df = drop_leakage_and_redundant(df)

    # 5. Sanity check
    sanity_check_processed(df)

    return df


def run_pipeline(
    raw_path: Path | str | None = None,
    output_path: Path = PROCESSED_PATH,
) -> pd.DataFrame:
    """End-to-end PH.02 pipeline:
      1. Load & validate raw CSV (Pandera)
      2. Engineer features + proxy target
      3. Persist to data/processed/
      4. Print summary statistics
    """
    print("=" * 60)
    print("PH.02 – Feature Engineering pipeline")
    print("=" * 60)

    # 1. Load + validate
    print("\n[1/4] Loading and validating raw data …")
    if raw_path is not None:
        raw = load_and_validate(raw_path)
    else:
        raw = load_and_validate()
    print(f"      Raw shape : {raw.shape}")
    print("      Pandera schema validation PASSED.")

    # 2. Feature engineering
    print("\n[2/4] Building features and regression target …")
    processed = build_features(raw)
    print(f"      Processed shape : {processed.shape}")
    print(f"      Final columns   : {processed.columns.tolist()}")

    # 3. Persist
    print("\n[3/4] Saving processed data …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"      Saved → {output_path}")

    # 4. Summary statistics for the proxy target
    print("\n[4/4] Proxy target summary (rul_proxy_min)")
    print("-" * 40)
    target = processed["rul_proxy_min"]
    print(f"  count  : {target.count():,}")
    print(f"  mean   : {target.mean():.2f}")
    print(f"  std    : {target.std():.2f}")
    print(f"  min    : {target.min()}")
    print(f"  25 %   : {target.quantile(0.25):.0f}")
    print(f"  50 %   : {target.median():.0f}")
    print(f"  75 %   : {target.quantile(0.75):.0f}")
    print(f"  max    : {target.max()}")
    print(f"  zeros  : {(target == 0).sum():,}  "
          f"(already at/past {TWF_RISK_WINDOW_START_MIN} min threshold)")

    print("\nDone.")
    return processed


if __name__ == "__main__":
    # Allow running from project root:  python src/features.py [raw_path] [out_path]
    raw_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else PROCESSED_PATH
    run_pipeline(raw_path=raw_arg, output_path=out_arg)