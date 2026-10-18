# Analyzing Customer Reviews for Personalized Skincare Recommendations

**Beauty & Wellness Data Science Internship — Weeks 1–6**  
**Author:** Rohit

---

## Project Objective

To determine whether product review text, star ratings, and user skin-profile information can support more useful skincare product recommendations than overall average star ratings alone.

**Research Questions (from Week 1):**
- RQ1: Can specific skin types and concerns be extracted from unstructured customer review text?
- RQ2: Is there a noticeable difference in how different skin profiles rate the same product?

**Hypothesis:** Average star ratings are misleading. Grouping reviews by skin type provides a more accurate recommendation signal.

---

## Data Source and License Notes

**Primary dataset:** Sephora Products and Skincare Reviews  
**Source:** Kaggle — https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews  
**License:** Not yet verified; check the Kaggle download page before redistribution or publication.

**Status:** CSV files are present in `data/raw/`. The full review corpus has been cleaned to `data/processed/reviews_clean.csv`; Week 4 EDA outputs are based on the documented 999-row sample.

> ⚠️ Do NOT commit raw data files to this repository unless the license explicitly permits redistribution.  
> ⚠️ Do NOT commit any credentials, API keys, or personal information.

---

## Folder Structure

```
/
├── Week_1_Report_Rohit.docx          # Week 1 submission (preserved)
├── README.md
├── requirements.txt
├── .gitignore
├── data_dictionary.md
├── checklist.md
├── data/
│   ├── raw/                          # Downloaded data (do not commit if license restricts)
│   └── processed/                    # Cleaned output from clean.py
├── reports/
│   ├── Week_2_Report.docx
│   ├── Week_3_Report.docx            # Week 3 cleaning report
│   ├── Week_4_Report.docx            # Week 4 sample EDA report
│   ├── Week_5_Report.docx            # Week 5 modeling report
│   └── Week_6_Report.docx            # To be created
├── src/
│   ├── clean.py                      # Week 3: cleaning script
│   ├── eda.py                        # Week 4: EDA and feature engineering
│   └── model.py                      # Week 5: modelling script
├── figures/                          # Charts saved by eda.py
└── tests/
    └── test_clean.py                 # Tests for cleaning rules
```

---

## Environment Setup

1. Ensure Python 3.9+ is installed: `python --version`
2. (Recommended) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Commands to Run

> All commands assume you are in the repository root and the virtual environment is active.

**Download data (manual step):**
1. Log in to Kaggle and download the Sephora dataset.
2. Place the CSV file(s) in `data/raw/`.
3. Record the license text and download date in `data_dictionary.md`.

**Run cleaning script (Week 3):**
```
python src/clean.py
```
Output: `data/processed/reviews_clean.csv` and `data/processed/audit_report.csv`.
The default run reads the numbered review CSV chunks and joins `primary_category`
from `product_info.csv` when it is not in the review files.

To run a bounded sample without replacing the existing processed files:
```
python src/clean.py --sample 1000 --output data/processed/reviews_clean_sample.csv --audit-output data/processed/audit_report_sample.csv
```
`--sample` reads the first N rows in file order; it does not load the full CSV first.

**Run EDA and feature engineering (Week 4):**
```
python src/eda.py
```
By default, EDA reads the actual Week 3 sample at
`data/processed/reviews_clean_sample.csv`. It writes supported charts to
`figures/`, descriptive feature data to
`data/processed/reviews_eda_features.csv`, and measured results to
`data/processed/eda_summary.json`, plus `reports/Week_4_Report.docx` with the
generated charts embedded. Existing outputs are not replaced unless
`--overwrite` is supplied.

To analyze a different cleaned dataset:
```
python src/eda.py --input data/processed/reviews_clean.csv --overwrite
```

**Week 4 tests:**
```
python -m pytest tests/test_eda.py -v
```

**Run modelling (Week 5):**
```
python src/model.py
```
The pipeline streams `data/processed/reviews_clean.csv`, takes a reproducible
30,000-row class-proportional sample, and holds out reviewers as groups. It
prints actual holdout and reviewer-grouped cross-validation metrics, and writes:

- `data/processed/model_summary.json`
- `figures/model_confusion_matrices.png`
- `figures/model_roc_curves.png`
- `reports/Week_5_Report.docx`

The pipeline refuses to replace existing Week 5 outputs unless `--overwrite` is
provided.

**Run tests:**
```
python -m pytest tests/ -v
```

---

## Expected Outputs

| Script | Output files |
|--------|-------------|
| `clean.py` | `data/processed/reviews_clean.csv`, `data/processed/audit_report.csv` |
| `eda.py` | `figures/rating_distribution.png`, `figures/missing_values.png`, `figures/review_length_distribution.png`, `figures/skin_type_distribution.png`, `figures/ratings_by_skin_type.png`, `figures/most_reviewed_products.png`, `data/processed/reviews_eda_features.csv`, `data/processed/eda_summary.json`, `reports/Week_4_Report.docx` |
| `model.py` | `data/processed/model_summary.json`, `figures/model_confusion_matrices.png`, `figures/model_roc_curves.png`, `reports/Week_5_Report.docx`, metrics printed to console |

---

## Known Limitations

- Dataset is a static 2023 snapshot from Sephora; findings may not generalise to other platforms.
- `skin_type` is self-reported by reviewers and may contain errors or missing values.
- Reviews are self-selected; rating distributions do not represent all purchasers.
- No skin_type data → skin-type comparisons will be skipped for rows with missing values.
- Results do not constitute medical or dermatological advice.

---

## Reproducibility Statement

All scripts use fixed random seeds where applicable. The cleaning, EDA, and modelling pipelines are fully scripted and can be re-run from scratch on the downloaded raw data. No results are hard-coded; all metrics are produced by executed code.

---

## Progress

- [x] Week 1: Problem definition (submitted, 44/100)
- [x] Week 2: Data sourcing report (`reports/Week_2_Report.docx`)
- [x] Week 3: Cleaning script, tests, sample outputs, full-corpus output, and report
- [x] Week 4: EDA script, figures, summary, and report (sample results; full corpus remains)
- [x] Week 5: Modelling script, tests, actual evaluation, figures, and report
- [ ] Week 6: Evaluation and business impact report
