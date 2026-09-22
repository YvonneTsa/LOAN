"""Evaluation helpers: metrics, confusion matrix, feature importance, tree plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import plot_tree


def classification_metrics(model, X, y) -> pd.DataFrame:
    """Accuracy, recall, precision and F1 for ``model`` on ``(X, y)``.

    Recall is the primary metric: the bank's costly error is a false negative,
    a customer who would accept a loan but is not targeted (missed revenue).
    """
    pred = model.predict(X)
    return pd.DataFrame(
        {
            "Accuracy": [accuracy_score(y, pred)],
            "Recall": [recall_score(y, pred)],
            "Precision": [precision_score(y, pred)],
            "F1": [f1_score(y, pred)],
        }
    )


def metrics_dict(model, X, y) -> dict[str, float]:
    """Metrics from :func:`classification_metrics` as a plain rounded dict."""
    return {k: round(float(v[0]), 4) for k, v in classification_metrics(model, X, y).to_dict("list").items()}


def plot_confusion_matrix(model, X, y, title="Confusion matrix", save_path: str | Path | None = None):
    """Plot a labelled confusion matrix (counts and percentages)."""
    cm = confusion_matrix(y, model.predict(X))
    labels = np.asarray([f"{c:0.0f}\n{c / cm.sum():.2%}" for c in cm.flatten()]).reshape(2, 2)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gca()


def feature_importance(model, feature_names) -> pd.Series:
    """Feature importances (Gini) as a sorted (descending) Series."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(model, feature_names, top_n: int | None = 15, save_path=None):
    """Horizontal bar plot of the most important features."""
    imp = feature_importance(model, feature_names)
    if top_n:
        imp = imp.head(top_n)
    imp = imp.sort_values()
    plt.figure(figsize=(10, max(5, len(imp) * 0.4)))
    plt.title("Feature importances")
    plt.barh(range(len(imp)), imp.values, color="teal", align="center")
    plt.yticks(range(len(imp)), imp.index)
    plt.xlabel("Relative importance (Gini)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gca()


def plot_decision_tree(model, feature_names, class_names=("No loan", "Loan"), save_path=None, figsize=(20, 12)):
    """Render the fitted decision tree."""
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=list(class_names),
        filled=True,
        rounded=True,
        fontsize=9,
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gca()
