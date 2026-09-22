# Data

The dataset `Loan_Modelling.csv` is included in this folder so the project runs out of the box.

The notebook and pipeline load it from here:

```python
loan_data = pd.read_csv('../data/Loan_Modelling.csv')   # notebook
# scripts/train.py reads the path from config.yaml
```

## Dataset summary

- **Rows:** 5,000 customers
- **Columns:** 14
- **Target:** `Personal_Loan` (1 = accepted the loan, 0 = did not); about 9.6% positive
- **Missing values:** none

## Data dictionary

| Column | Description |
| --- | --- |
| `ID` | Customer ID (dropped before modeling) |
| `Age` | Customer age in years |
| `Experience` | Years of professional experience |
| `Income` | Annual income (in thousands of USD) |
| `ZIPCode` | Home address ZIP code |
| `Family` | Family size |
| `CCAvg` | Average monthly credit-card spending (in thousands of USD) |
| `Education` | Education level: 1 = Undergraduate, 2 = Graduate, 3 = Advanced/Professional |
| `Mortgage` | Value of house mortgage, if any (in thousands of USD) |
| `Personal_Loan` | **Target**: did the customer accept the personal loan in the last campaign? |
| `Securities_Account` | Does the customer have a securities account with the bank? |
| `CD_Account` | Does the customer have a certificate-of-deposit account? |
| `Online` | Does the customer use online banking? |
| `CreditCard` | Does the customer use a credit card issued by another bank? |

## Data-quality notes handled in the notebook

- `ID` is a unique identifier with no predictive value and is dropped.
- `Experience` contains a few negative values (data-entry errors); -1, -2, -3 are mapped to 1, 2, 3.
- `ZIPCode` has 467 unique values. It is reduced to its first two digits (the geographic region) and treated as a category rather than a number.
- `Experience` is dropped from the model features because it is almost perfectly correlated with `Age`.
