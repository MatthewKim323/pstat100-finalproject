# PSTAT 100 Final Project — Predicting Income from Census Data

Predicting whether a U.S. adult earns **more than \$50,000/year** from demographic
and employment features, using the **UCI Adult / Census Income** dataset (Becker &
Kohavi, 1994 Census extract).

**Group:** Ethan Chuang, Matthew Kim, Edward Lee, Angela Pei, Matthew Tran

## What's here

- **`Final_Project_Report.ipynb`** — the full reproducible report: abstract,
  introduction, methodology, data preparation & EDA, results, conclusion, and
  references.
- **`database/`** — the UCI Adult distribution (`adult.data`, `adult.test`,
  `adult.names`) used by the notebook.

## Summary

We treat the task as binary classification on the official train/test split. After
cleaning missing tokens (`?` → `NaN`), checking duplicates, and exploring the data,
we fit a **logistic regression** and a **random forest** inside a single `sklearn`
pipeline (median/most-frequent imputation, scaling, one-hot encoding).

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Majority-class baseline | 0.764 | 0.500 |
| Logistic regression | 0.851 | 0.904 |
| Random forest | 0.851 | 0.901 |

Both models beat the majority baseline by ~9 percentage points; the logistic model
holds a slight edge on ROC-AUC. Education and hours worked separate the income
groups most clearly. Results are **associational** (a 1994 historical extract),
not causal.

## Reproducing

```bash
pip install numpy pandas matplotlib seaborn missingno scikit-learn jupyter
jupyter notebook  # open Final_Project_Report.ipynb from this folder
```

Run the notebook top-to-bottom from this directory so that `database/adult.data`
resolves correctly.

## References

- Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5XW20
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12.
