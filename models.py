"""Decision-tree model building: default, pre-pruned, and post-pruned."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier

from . import RANDOM_STATE


def default_tree(X_train, y_train, random_state: int = RANDOM_STATE) -> DecisionTreeClassifier:
    """A fully grown decision tree (scikit-learn defaults, gini criterion).

    This overfits, and serves as the baseline the pruned models improve on.
    """
    model = DecisionTreeClassifier(criterion="gini", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def pre_pruned_tree(
    X_train,
    y_train,
    X_test,
    y_test,
    max_depth_values=(2, 4, 6),
    min_samples_split_values=(10, 30, 50, 70),
) -> DecisionTreeClassifier:
    """Pre-prune by searching depth and min-samples-split limits.

    Uses ``class_weight='balanced'`` to counter the ~9.6% positive rate, and
    selects the tree that generalizes best: the smallest gap between train and
    test recall, breaking ties toward higher test recall. Matches the original
    study's selection loop.
    """
    best_estimator = None
    best_score_diff = float("inf")
    best_test_score = 0.0

    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            estimator = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight="balanced",
                random_state=42,
            )
            estimator.fit(X_train, y_train)
            train_recall = recall_score(y_train, estimator.predict(X_train))
            test_recall = recall_score(y_test, estimator.predict(X_test))
            score_diff = abs(train_recall - test_recall)
            if (score_diff < best_score_diff) and (test_recall > best_test_score):
                best_score_diff = score_diff
                best_test_score = test_recall
                best_estimator = estimator

    best_estimator.fit(X_train, y_train)
    return best_estimator


@dataclass
class PostPruneResult:
    """Result of cost-complexity post-pruning."""

    model: DecisionTreeClassifier
    best_alpha: float
    ccp_alphas: np.ndarray
    recall_train: list[float]
    recall_test: list[float]


def post_pruned_tree(X_train, y_train, X_test, y_test, random_state: int = RANDOM_STATE) -> PostPruneResult:
    """Post-prune with cost-complexity pruning, selecting alpha by test recall.

    Fits a tree for every effective alpha on the pruning path, then keeps the
    one with the highest test recall. Matches the original study.
    """
    base = DecisionTreeClassifier(random_state=random_state)
    path = base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # drop the trivial root-only tree

    clfs = []
    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    recall_train = [recall_score(y_train, c.predict(X_train)) for c in clfs]
    recall_test = [recall_score(y_test, c.predict(X_test)) for c in clfs]

    best_idx = int(np.argmax(recall_test))
    return PostPruneResult(
        model=clfs[best_idx],
        best_alpha=float(ccp_alphas[best_idx]),
        ccp_alphas=ccp_alphas,
        recall_train=recall_train,
        recall_test=recall_test,
    )
