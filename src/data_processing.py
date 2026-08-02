"""
data_processing.py
------------------
Phase 2 (PH.02) — Loading and schema validation for the raw AI4I 2020 dataset.

Responsibility of this module (single responsibility, per project architecture):
    1. Load the raw CSV exactly as downloaded from UCI.
    2. Validate it against a strict Pandera schema *immediately* after loading,
       before any feature engineering happens.

Why validate here and not later:
    Pandera is the guardrail for the *file on disk* (data/raw/ai4i2020.csv) — it
    answers "is this still the dataset I think it is?". Pydantic (used later in
    api/schemas.py) guards a different boundary: a single user-typed prediction
    request. If the raw CSV is ever replaced, edited, or corrupted, this schema
    stops the pipeline right here with a clear error, instead of failing silently
    three steps later inside feature engineering or training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from pandera.pandas import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Schema for the RAW file, as published by UCI (before any feature engineering)
# ---------------------------------------------------------------------------
# Ranges are chosen from the dataset's documented generative process
# (DatasetDoc.txt): air temp ~300 K +/- 2 K, process temp = air temp + 10 K +/- 1 K,
# torque ~40 Nm +/- 10 Nm with no negative values, tool wear in minutes.
# The bounds below are deliberately generous (not just +/- a few sigma) so that
# natural variation isn't flagged as invalid, while still catching real
# corruption (e.g. a negative Kelvin temperature, a non-binary failure flag).
# ---------------------------------------------------------------------------
RAW_SCHEMA = DataFrameSchema(
    {
        "UDI": Column(int, checks=Check.greater_than(0), unique=True, nullable=False),
        "Product ID": Column(str, nullable=False),
        "Type": Column(str, checks=Check.isin(["L", "M", "H"]), nullable=False),
        "Air temperature [K]": Column(
            float, checks=Check.in_range(250.0, 350.0), nullable=False
        ),
        "Process temperature [K]": Column(
            float, checks=Check.in_range(250.0, 350.0), nullable=False
        ),
        "Rotational speed [rpm]": Column(
            int, checks=Check.greater_than(0), nullable=False
        ),
        "Torque [Nm]": Column(
            float, checks=Check.greater_than_or_equal_to(0.0), nullable=False
        ),
        "Tool wear [min]": Column(
            int, checks=Check.greater_than_or_equal_to(0), nullable=False
        ),
        "Machine failure": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "TWF": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "HDF": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "PWF": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "OSF": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "RNF": Column(int, checks=Check.isin([0, 1]), nullable=False),
    },
    checks=[
        Check(
            lambda df: (
                df["Process temperature [K]"] - df["Air temperature [K]"]
            ).between(2, 18).all(),
            error="Process temperature is not within the documented ~10 K offset band above air temperature.",
        )
    ],
    strict=False,  # allow extra columns without failing (forward-compatible)
    coerce=True,   # cast dtypes (e.g. int64 vs int32) instead of failing on them
)


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw AI4I 2020 CSV from disk.

    Parameters
    ----------
    path : str or Path
        Path to the raw CSV file (default matches the project's folder layout).

    Returns
    -------
    pd.DataFrame
        The raw, unvalidated dataframe exactly as read from disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    # encoding="utf-8-sig" handles the occasional UTF-8 BOM that some CSV exporters prepend
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw dataframe against RAW_SCHEMA.

    Raises
    ------
    pandera.errors.SchemaErrors
        If the dataframe does not match the expected columns, dtypes, or
        value ranges (lazy=True collects *all* violations at once).
    pandera.errors.SchemaError
        Fallback for older Pandera versions that raise single errors.

    Returns
    -------
    pd.DataFrame
        The (possibly coerced) validated DataFrame.
    """
    return RAW_SCHEMA.validate(df, lazy=True)


def load_and_validate(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load the raw CSV and validate it in one call.

    This is the function every other module (features.py, train.py) should
    import, so the schema check can never accidentally be skipped.
    """
    df = load_raw(path)
    df = validate_raw(df)
    return df


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else RAW_DATA_PATH
    try:
        data = load_and_validate(raw_path)
        print(f"Loaded and validated {len(data):,} rows from {raw_path}")
        print(f"Columns: {list(data.columns)}")
        print("Schema validation PASSED.")
    except FileNotFoundError as exc:
        print(f"File not found: {exc}")
        print("Place the raw ai4i2020.csv under data/raw/ before running this script.")
        sys.exit(1)
    except SchemaErrors as exc:
        print("Schema validation FAILED — the raw file does not match the expected shape/ranges.")
        print(exc)
        sys.exit(1)
    except SchemaError as exc:
        # Fallback for non-lazy single-error raises (older pandera versions)
        print("Schema validation FAILED — the raw file does not match the expected shape/ranges.")
        print(exc)
        sys.exit(1)