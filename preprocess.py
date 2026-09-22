"""Feature construction and train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from . import RANDOM_STATE, TARGET

# Experience is dropped: it is almost perfectly correlated with Age.
DROP_FROM_FEATURES = [TARGET, "Experience"]
DUMMY_COLUMNS = ["ZIPCode", "Education", "Family"]


@dataclass
class DataSplits:
    """Container for the encoded feature matrix and its train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split off the target and one-hot encode the categorical predictors.

    Drops ``Experience`` (collinear with ``Age``) and dummy-encodes
    ``ZIPCode``, ``Education`` and ``Family`` with ``drop_first=True``.
    """
    y = df[TARGET]
    X = df.drop(columns=DROP_FROM_FEATURES)
    X = pd.get_dummies(X, columns=DUMMY_COLUMNS, drop_first=True)
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Create a 70/30 train/test split.

    Matches the original study, which used a plain (non-stratified) split at
    ``random_state=1`` so results reproduce exactly.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return DataSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )
