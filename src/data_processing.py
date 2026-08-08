"""
data_processing.py
-------------------
Loading and schema validation for the raw AI4I 2020 dataset.

Responsibilities of this module
--------------------------------
    1. Load the raw CSV exactly as downloaded from UCI, without modification.
    2. Validate it against a strict Pandera schema immediately after loading,
       before any feature engineering is applied.

Why validation happens here, not later
-----------------------------------------
Pandera guards the boundary of the file on disk (data/raw/ai4i2020.csv) —
it answers the question "is this still the dataset I think it is?". This is
a distinct concern from validating a single user-submitted prediction
request at inference time, which belongs to a different layer and checks
one row against business rules rather than a full dataset against its
expected distribution.

If the raw CSV is ever replaced, edited, or corrupted, this schema stops
the pipeline immediately with a clear, itemised error — instead of failing
silently several steps later, inside feature engineering or model training,
where the root cause would be far harder to trace.
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
# Value ranges follow the dataset's documented generative process
# (DatasetDocs.txt): air temperature ~300 K +/- 2 K, process temperature =
# air temperature + 10 K +/- 1 K, torque ~40 Nm +/- 10 Nm with no negative
# values, tool wear expressed in whole minutes.
#
# Bounds are set deliberately wide (not just a few standard deviations) so
# that normal sampling variation is never flagged as invalid, while still
# catching genuine corruption — e.g. a negative Kelvin temperature or a
# non-binary failure flag.
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
    """Load the raw AI4I 2020 CSV from disk, unmodified.

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
        value ranges (lazy=True collects all violations at once, rather
        than stopping at the first one).
    pandera.errors.SchemaError
        Fallback for older Pandera versions that raise a single error
        instead of an aggregated SchemaErrors collection.

    Returns
    -------
    pd.DataFrame
        The validated dataframe, with dtypes coerced where the schema
        requests it.
    """
    return RAW_SCHEMA.validate(df, lazy=True)


def load_and_validate(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load the raw CSV and validate it in one call.

    This is the function every other module in the pipeline (e.g. features.py,
    train.py) should import, so schema validation can never be accidentally
    skipped by calling load_raw() directly.
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