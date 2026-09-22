# Resume & Portfolio Bullet Points

Copy-ready lines describing this project, tailored for data-analytics and fintech roles. Pick the version that fits the space you have.

## Why this project is relevant to fintech

Personal-loan response is a core fintech decisioning problem: an imbalanced target (about 9.6% acceptance), a cost trade-off between over-targeting and missing good prospects, and a need for interpretable drivers a business team can act on. The same shape appears in credit targeting, cross-sell, churn prevention, and campaign optimization. This project uses decision trees precisely because they are transparent: every prediction traces to a readable rule.

## Resume (concise)

- Built and tuned decision-tree models to predict personal-loan acceptance for a bank marketing campaign on 5,000 customers, reaching 92% recall and 88% precision on a held-out test set.
- Handled class imbalance and compared default, pre-pruned (balanced class weights), and cost-complexity post-pruned trees, rejecting a degenerate perfect-recall model in favor of the best balanced F1.
- Identified income, education, and family size as the top drivers of acceptance and translated them into concrete campaign-targeting recommendations.
- Shipped as a reproducible, production-style repository: modular Python package, one-command training pipeline, persisted model, and an executive report.

## Resume (one-liner)

- Developed an interpretable decision-tree model for bank loan-campaign targeting (92% recall, 88% precision), shipped as a reproducible, documented pipeline.

## Portfolio / LinkedIn (expanded)

**Personal Loan Prediction (AllLife Bank campaign)**
Built an end-to-end classification project to predict which liability customers will accept a personal loan, framed as an imbalanced marketing-response problem common in fintech. Cleaned and engineered the data (region from ZIP code, corrected data-entry anomalies), ran full EDA, then built and compared three decision trees: a default baseline, a pre-pruned tree with balanced class weights, and a cost-complexity post-pruned tree. Selected on balanced F1 rather than raw recall, rejecting a perfect-recall model that flagged nearly every customer. The final model reached 92% recall and 88% precision on the test set, with income, education, and family size as the dominant drivers. Delivered as a reproducible repository with a modular package, a one-command pipeline, a saved model, and a written report.

**Stack:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib

## Skills demonstrated

Data cleaning, feature engineering, exploratory data analysis, handling class imbalance, decision trees, pre-pruning and cost-complexity post-pruning, cost-sensitive metric selection, model interpretability, reproducible ML engineering, business insight communication
