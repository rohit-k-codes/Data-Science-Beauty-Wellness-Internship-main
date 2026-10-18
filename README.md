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

**Status:** The full review corpus has been cleaned to `data/processed/reviews_clean.csv` (969,885 rows). Week 4 EDA outputs and `Week_4_Report.docx` use the documented 999-row, first-rows sample; they should not be treated as representative of the full corpus.

> ⚠️ Do NOT commit raw data files to this repository unless the license explicitly permits redistribution.  
> ⚠️ Do NOT commit any credentials, API keys, or personal information.

---

## Folder Structure

```
/
├── README.md
├── requirements.txt
├── .gitignore
├── checklist.md
├── data_dictionary.md
├── data/
│   ├── raw/                          # Downloaded data (do not commit if license restricts)
│   └── processed/                    # Cleaned output from clean.py
├── reports/
│   ├── Week_1_Report.docx            # Problem definition
│   ├── Week_2_Report.docx
│   ├── Week_3_Report.docx            # Week 3 cleaning report
│   ├── Week_4_Report.docx            # Week 4 sample EDA report
│   ├── Week_5_Report.docx            # Week 5 modeling report
│   ├── Week_6_Report.docx            # Final evaluation and business impact
│   └── Final_Project_Summary.docx    # Consolidated Weeks 1–6 summary
├── src/
│   ├── clean.py                      # Week 3: cleaning script
│   ├── eda.py                        # Week 4: EDA and feature engineering
│   └── model.py                      # Week 5: modeling script
├── scripts/
│   └── make_sample.py                # Synthetic edge-case CSV generator; not real data
├── figures/                          # EDA and Week 5 evaluation charts
└── tests/
    ├── test_clean.py
    ├── test_eda.py
    └── test_model.py
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
This default command writes to those paths; use distinct `--output` and
`--audit-output` paths if you need to preserve an existing local result.
The default run reads the numbered review CSV chunks and joins `primary_category`
from `product_info.csv` when it is not in the review files.

To run a bounded sample without replacing the existing processed files:
```
python src/clean.py --sample 1000 --output data/processed/reviews_clean_verify_sample.csv --audit-output data/processed/audit_report_verify_sample.csv
```
`--sample` reads the first N rows in file order; it does not load the full CSV first.
The cleaner writes directly to the requested paths, so use new names if those
files already exist.

**Run EDA and feature engineering (Week 4):**
```
python src/eda.py --input data/processed/reviews_clean.csv --figures-dir figures/full_corpus --summary data/processed/eda_summary_full.json --features data/processed/reviews_eda_features_full.csv --report reports/Week_4_Report_Full.docx
```
The command above uses separate output paths so the existing 999-row sample
charts and report remain intact. EDA loads the selected CSV into memory; allow
adequate memory for the full corpus.

The default EDA input is `data/processed/reviews_clean_sample.csv`. On a fresh
run it writes six supported charts to `figures/`, feature data to
`data/processed/reviews_eda_features.csv`, measured results to
`data/processed/eda_summary.json`, and an embedded-chart report to
`reports/Week_4_Report.docx`. It refuses to replace existing outputs unless
`--overwrite` is supplied.

**Week 4 tests:**
```
python -m pytest tests/test_eda.py -v
```

**Run modelling (Week 5):**
```
python src/model.py --summary data/processed/model_summary_rerun.json --confusion-plot figures/model_confusion_matrices_rerun.png --roc-plot figures/model_roc_curves_rerun.png --report reports/Week_5_Report_Rerun.docx
```
The pipeline streams `data/processed/reviews_clean.csv`, takes a reproducible
30,000-row class-proportional sample, and holds out reviewers as groups. It
prints actual holdout and reviewer-grouped cross-validation metrics, and writes:

- `data/processed/model_summary.json`
- `figures/model_confusion_matrices.png`
- `figures/model_roc_curves.png`
- `reports/Week_5_Report.docx`

Use unique output paths as shown to preserve existing Week 5 files. The
pipeline refuses to replace existing outputs unless `--overwrite` is provided;
that flag replaces the selected summary, charts, and report. Fitted estimators
are trained in memory and are not serialized.

**Run tests:**
```
python -m pytest tests/ -v
```

---

## Expected Outputs

| Script | Output files |
|--------|-------------|
| `clean.py` | `data/processed/reviews_clean.csv`, `data/processed/audit_report.csv` |
| `eda.py` | Six charts (rating, missingness, review length, skin type, ratings by skin type, products), feature CSV, JSON summary, and DOCX report; paths are configurable |
| `model.py` | `data/processed/model_summary.json`, `figures/model_confusion_matrices.png`, `figures/model_roc_curves.png`, `reports/Week_5_Report.docx`, metrics printed to console |

---

## Known Limitations

- Dataset is a static 2023 snapshot from Sephora; findings may not generalise to other platforms.
- `skin_type` is self-reported by reviewers and may contain errors or missing values.
- Reviews are self-selected; rating distributions do not represent all purchasers.
- Week 4 EDA covers only a 999-row prefix sample; 998 records are for one product, so its profile comparisons are not representative product-level evidence.
- The Week 5 target is a historical review-level `is_recommended` flag. Review text can express sentiment, and rating is strongly associated with the flag; the model is not a pre-purchase product recommender.
- No validated skin-concern extractor, product-level statistical test, incremental skin-profile ablation, calibration study, or real-user business-impact experiment was completed.
- The exact dataset license and download date remain unverified. Do not redistribute or commit raw or processed data until the license is confirmed.
- Results do not constitute medical or dermatological advice.

---

## Reproducibility Statement

All scripts use fixed random seeds where applicable. The cleaning, EDA, and modelling pipelines are fully scripted and can be re-run from scratch on the downloaded raw data. No results are hard-coded; all metrics are produced by executed code.

---

## Progress

- [x] Week 1: Problem definition (submitted, 44/100)
- [x] Week 2: Data sourcing report (`reports/Week_2_Report.docx`)
- [x] Week 3: Cleaning script, tests, full-corpus output, audit, and report
- [x] Week 4: EDA script, tests, six figures, sample summary, and sample report
- [x] Week 5: Modeling pipeline, tests, reproducible evaluation, figures, and report
- [x] Week 6: Final evaluation report, consolidated project summary, and checklists
- [ ] Before submission: verify and record the dataset license and download date
- [ ] Recommended before claiming personalization: rerun full-corpus EDA and perform product-grouped, adequately powered profile comparisons
