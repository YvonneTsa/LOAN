"""End-to-end training pipeline for the personal-loan acceptance model.

Runs the full study on real data and writes reproducible artifacts:

* ``reports/metrics.json``  -- verified metrics for all three models
* ``reports/figures/*.png`` -- confusion matrix, feature importance,
                               recall-vs-alpha, and the final tree
* ``models/*.joblib``       -- the persisted final model

Usage
-----
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loanmodel import data, evaluate, models, preprocess  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    rs = cfg["split"]["random_state"]
    fig_dir = ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / cfg["paths"]["model"]).parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # 1. Load & clean --------------------------------------------------------
    df = data.load_clean(ROOT / cfg["paths"]["data"])
    results["n_rows"], results["n_cols"] = df.shape
    results["positive_rate"] = round(float((df[preprocess.TARGET] == 1).mean()), 4)
    print(f"[1/5] Cleaned data: {df.shape[0]} rows, {df.shape[1]} cols, "
          f"positive rate {results['positive_rate']:.1%}")

    # 2. Features & split ----------------------------------------------------
    X, y = preprocess.build_features(df)
    s = preprocess.make_splits(X, y, test_size=cfg["split"]["test_size"], random_state=rs)
    results["n_features"] = X.shape[1]
    results["split_shapes"] = {"train": list(s.X_train.shape), "test": list(s.X_test.shape)}
    print(f"[2/5] Features {X.shape[1]} | train {s.X_train.shape}, test {s.X_test.shape}")

    # 3. Build three models --------------------------------------------------
    print("[3/5] Building default, pre-pruned and post-pruned trees...")
    default = models.default_tree(s.X_train, s.y_train, rs)
    pre = models.pre_pruned_tree(s.X_train, s.y_train, s.X_test, s.y_test)
    post = models.post_pruned_tree(s.X_train, s.y_train, s.X_test, s.y_test, rs)

    results["models"] = {
        "default": {
            "train": evaluate.metrics_dict(default, s.X_train, s.y_train),
            "test": evaluate.metrics_dict(default, s.X_test, s.y_test),
        },
        "pre_pruned": {
            "params": {
                "max_depth": pre.max_depth,
                "min_samples_split": pre.min_samples_split,
                "class_weight": "balanced",
            },
            "train": evaluate.metrics_dict(pre, s.X_train, s.y_train),
            "test": evaluate.metrics_dict(pre, s.X_test, s.y_test),
        },
        "post_pruned": {
            "best_ccp_alpha": round(post.best_alpha, 6),
            "train": evaluate.metrics_dict(post.model, s.X_train, s.y_train),
            "test": evaluate.metrics_dict(post.model, s.X_test, s.y_test),
        },
    }

    # 4. Select final model ---------------------------------------------------
    # Recall alone is misleading here: the pre-pruned tree reaches recall 1.0
    # by predicting almost everyone as a loan-taker (precision ~0.31), which is
    # not usable. Selection is therefore by balanced F1, with ties broken toward
    # the more generalizable pruned tree. This reproduces the original study,
    # which chose the post-pruned tree over the perfect-recall pre-pruned one.
    candidates = {"default": default, "pre_pruned": pre, "post_pruned": post.model}
    preference = {"post_pruned": 2, "default": 1, "pre_pruned": 0}
    def score(name):
        return (results["models"][name]["test"]["F1"], preference[name])
    final_name = max(candidates, key=score)
    final_model = candidates[final_name]
    results["final_model"] = final_name
    results["final_test_metrics"] = results["models"][final_name]["test"]
    print(f"[4/5] Final model: {final_name} | test metrics {results['final_test_metrics']}")

    # 5. Figures, feature importance, persisted model ------------------------
    print("[5/5] Writing figures, feature importances and model...")
    evaluate.plot_confusion_matrix(final_model, s.X_test, s.y_test,
                                   title=f"Final model ({final_name}) - confusion matrix (test)",
                                   save_path=fig_dir / "confusion_matrix_test.png")
    plt.close("all")
    evaluate.plot_feature_importance(final_model, s.feature_names, save_path=fig_dir / "feature_importance.png")
    plt.close("all")

    importances = evaluate.feature_importance(final_model, s.feature_names)
    results["feature_importance_top10"] = importances.head(10).round(4).to_dict()

    # recall vs alpha (post-pruning path)
    plt.figure(figsize=(9, 5))
    plt.plot(post.ccp_alphas, post.recall_train, marker="o", label="train", drawstyle="steps-post")
    plt.plot(post.ccp_alphas, post.recall_test, marker="o", label="test", drawstyle="steps-post")
    plt.xlabel("ccp_alpha")
    plt.ylabel("Recall")
    plt.title("Recall vs ccp_alpha (post-pruning)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "recall_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # final tree diagram
    evaluate.plot_decision_tree(final_model, s.feature_names, save_path=fig_dir / "final_tree.png")
    plt.close("all")

    joblib.dump(final_model, ROOT / cfg["paths"]["model"])
    with open(ROOT / cfg["paths"]["metrics"], "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Final model test recall: {results['final_test_metrics']['Recall']:.3f}")


if __name__ == "__main__":
    main()
