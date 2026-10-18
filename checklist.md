# Project Checklist

**Project:** Analyzing Customer Reviews for Personalized Skincare Recommendations  
**Intern:** Rohit

---

## Week 2

- [x] Week 2 sourcing report, comparison, planned analysis, ethics, limitations, and references are present.
- [x] Sephora review CSV files are present and were read by the cleaning pipeline.
- [ ] **BLOCKED:** Verify the exact dataset license and record the license text and download date in `data_dictionary.md`.

## Week 3

- [x] Cleaning pipeline, tests, audit, sample output, full cleaned output, and report are present.
- [x] Existing full-corpus audit records 1,094,411 raw rows and 969,885 cleaned rows.
- [x] Final verification ran `python src/clean.py --sample 1000` with separate temporary outputs; it completed with 999 cleaned rows.
- [x] Raw inputs are unchanged and ignored by Git.
- [ ] The full cleaner was not rerun in final verification to avoid replacing existing outputs; rerun using distinct output paths if a fresh full rebuild is required.

## Week 4

- [x] EDA script, tests, six chart files, sample summary, derived features, and sample report are present.
- [x] Final verification reran EDA against the 999-row sample and generated all six charts to temporary output paths.
- [x] Existing summary reports 999 rows, 24 source columns, and six missing-value columns.
- [ ] Full-corpus EDA and report have not been generated. The current first-rows sample has 998 of 999 rows from a single product, so its profile comparisons are not representative product-level evidence.

## Week 5

- [x] Modeling pipeline, tests, machine-readable results, evaluation plots, and report are present.
- [x] Final verification reran the complete modeling experiment; saved and fresh run results match.
- [x] Majority-class baseline, Logistic Regression, and Decision Tree were compared.
- [x] Reviewer-grouped holdout and three-fold GroupKFold were used; reviewer overlap across the holdout is zero.
- [x] Rating and rating-derived fields were excluded from predictors; model preprocessing is fitted inside each pipeline.
- [ ] Fitted estimators are not serialized; `src/model.py` retrains them on each run.

## Week 6

- [x] `reports/Week_6_Report.docx` contains verified results, figures, business implications, limitations, conclusion, and future scope.
- [x] `reports/Final_Project_Summary.docx` consolidates Weeks 1–6 and includes reproducibility instructions.
- [x] Model comparison, confusion counts, privacy/fairness considerations, uncertainty, and medical-use limits are documented.
- [x] No unmeasured sales, conversion, return, trust, or other business-impact claims are made.
- [x] Full automated suite passed: 76 tests.

## Final Submission Checklist

- [x] Week 1–5 reports and the expected scripts/tests are present.
- [x] Relevant charts exist; DOCX files opened and embedded images/text were checked with `python-docx`.
- [x] Week 5 report metrics match the latest successful modeling run.
- [x] `data/raw/` and `data/processed/` are ignored by Git; no data files are tracked.
- [x] Common credential patterns were not found in tracked text files.
- [ ] **BLOCKED:** Confirm licensing and record the exact license text and download date before submission or redistribution.
- [ ] Open both new DOCX files in Microsoft Word and inspect page layout; Word GUI rendering was not available during verification.
- [ ] If the evaluator expects a supported RQ2 conclusion, run full-corpus EDA in separate paths and conduct adequately powered per-product profile comparisons with appropriate statistical testing.
- [ ] Confirm the repository URL and any additional internship submission requirements with the evaluator.
- [ ] Keep raw/processed data out of public commits unless the dataset license explicitly permits redistribution.
