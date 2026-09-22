"""Data loading and cleaning for the AllLife Bank personal-loan dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw Loan_Modelling CSV unchanged."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe, applying the fixes established during EDA.

    * ``ID`` is a unique identifier with no predictive value and is dropped.
    * ``Experience`` has a few negative values (data-entry errors); the
      known cases (-1, -2, -3) are mapped to their positive counterparts.
    * ``ZIPCode`` is a high-cardinality identifier (467 unique values). It is
      reduced to its first two digits (the geographic region) and treated as
      a category rather than a number.
    * ``Education``, ``ZIPCode`` and ``Family`` are cast to categorical.

    The input is not mutated; a cleaned copy is returned.
    """
    df = df.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df["Experience"] = df["Experience"].replace({-1: 1, -2: 2, -3: 3})

    df["ZIPCode"] = df["ZIPCode"].astype(str).str[0:2].astype("category")

    for col in ("Education", "ZIPCode", "Family"):
        df[col] = df[col].astype("category")

    return df


def load_clean(path: str | Path) -> pd.DataFrame:
    """Convenience wrapper: :func:`load_raw` followed by :func:`clean`."""
    return clean(load_raw(path))
