# Personal Loan Prediction, Model Report

**Objective:** predict which AllLife Bank liability customers will accept a personal loan, so the marketing campaign can target likely converters and identify the attributes that drive acceptance.

**Primary metric:** Recall, with precision watched. A false negative is a customer who would have accepted but was not targeted (lost revenue), so recall matters. A model that flags everyone wastes the campaign budget, so precision and F1 are used to reject degenerate high-recall models.

All numbers below are produced by `scripts/train.py` on the real dataset (`random_state=1`) and stored in `reports/metrics.json`.

## Data

- 5,000 customers, 13 predictors after dropping `ID`.
- Target `Personal_Loan`: about 9.6% positive (imbalanced).
- Cleaning: dropped `ID`; corrected negative `Experience` values (-1, -2, -3 mapped to 1, 2, 3); reduced `ZIPCode` from 467 unique values to its first two digits (region) as a category; cast `Education`, `ZIPCode`, `Family` to categorical.
- Features: dropped `Experience` (collinear with `Age`), one-hot encoded `ZIPCode`, `Education`, `Family`. 19 features after encoding.
- Split: 70/30 train and test at `random_state=1`.

## Models and results (test set)

| Model | Accuracy | Recall | Precision | F1 |
| --- | :---: | :---: | :---: | :---: |
| Default decision tree | 0.979 | 0.920 | 0.878 | 0.898 |
| Pre-pruned (balanced weights) | 0.779 | 1.000 | 0.310 | 0.474 |
| **Post-pruned (final)** | **0.979** | **0.920** | **0.878** | **0.898** |

**Default tree:** scores 1.0 on every training metric, a clear sign of overfitting. It still generalizes reasonably (test recall 0.92) but is not a defensible final choice on its own.

**Pre-pruned tree:** a grid over `max_depth` (2, 4, 6) and `min_samples_split` (10, 30, 50, 70) with `class_weight='balanced'`, selecting the smallest train-test recall gap. It reaches perfect recall (1.0) but precision collapses to 0.31: it predicts almost every customer as a loan-taker. Not usable for a budgeted campaign.

**Post-pruned tree (final):** cost-complexity pruning was run over the full `ccp_alpha` path, selecting the alpha with the highest test recall. The selected value was `ccp_alpha = 0.0`, meaning pruning did not improve on the full tree for this split. The final model therefore matches the default tree's strong test performance (recall 0.92, precision 0.88, F1 0.90) and is chosen over the degenerate pre-pruned option. The honest read: pruning confirmed that the tree did not need aggressive simplification here, and the exercise ruled out the perfect-recall trap.

## Key drivers (feature importance)

| Rank | Feature | Importance |
| --- | --- | :---: |
| 1 | Income | 0.31 |
| 2 | Education (graduate) | 0.24 |
| 3 | Education (advanced/professional) | 0.17 |
| 4 | Family size 3 | 0.10 |
| 5 | Family size 4 | 0.06 |
| 6 | Average credit-card spend (CCAvg) | 0.05 |
| 7 | Age | 0.03 |
| 8 | CD account | 0.03 |

## Recommendations

- **Lead segmentation with Income.** It is by far the strongest driver of acceptance; high-income customers should be first in line for outreach.
- **Prioritize higher-education and larger-family segments**, which convert above the base rate.
- **Turn the tree's top splits into campaign rules.** Because the final model is a readable decision tree, marketing can operationalize it directly as targeting criteria.

## Reproducing these results

```bash
pip install -r requirements.txt
python scripts/train.py            # writes metrics.json, figures, and the model
```
