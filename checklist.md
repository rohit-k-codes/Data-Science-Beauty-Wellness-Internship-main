# Project Checklist

**Project:** Analyzing Customer Reviews for Personalized Skincare Recommendations  
**Intern:** Rohit

---

## Before Submitting Each Week

- [ ] Report .docx file opens without errors in Microsoft Word
- [ ] All referenced figures exist at the stated path
- [ ] No placeholder text (e.g. "TODO", "INSERT HERE", "TBD") remains in any report
- [ ] No fabricated metrics or numbers have been added
- [ ] All scripts run without errors (python src/clean.py, python src/eda.py, python src/model.py)
- [ ] Tests pass: python -m pytest tests/ -v
- [ ] README.md is up to date with current folder structure and commands
- [ ] No credentials, API keys, or personal data are in any committed file

---

## Week 2 Checklist

- [x] reports/Week_2_Report.docx created and validates
- [x] Five candidate data sources compared in a structured table
- [x] Dataset selection criteria defined (Must-Have vs Nice-to-Have)
- [x] Recommended source justified with rationale
- [x] Source log completed with check dates
- [x] Data dictionary defined for all expected fields
- [x] Planned statistics described (clearly labelled as planned, not executed)
- [x] Proposed charts listed with variables and research questions addressed
- [x] How data tests Week 1 hypotheses documented
- [x] Ethics, licensing, and limitations section included
- [x] References with URLs included
- [x] INTERN ACTION: Download Sephora dataset from Kaggle
- [ ] INTERN ACTION: Record exact license text in data_dictionary.md
- [ ] INTERN ACTION: Record download date in data_dictionary.md
- [x] INTERN ACTION: Confirm files open correctly (pd.read_csv returns expected columns)

---

## Week 3 Checklist

- [x] reports/Week_3_Report.docx created
- [x] src/clean.py script created and runs without errors
- [x] Raw data preserved in data/raw/ (not modified)
- [x] Cleaned 1,000-row actual-data sample saved to data/processed/reviews_clean_sample.csv
- [x] Row count logged before and after each cleaning rule (audit_report_sample.csv)
- [x] Missing skin_type treated as unknown (not imputed)
- [x] tests/test_clean.py created and all 64 tests pass
- [x] scripts/make_sample.py generates synthetic sample for testing without real data
- [ ] INTERN ACTION: Run python src/clean.py on the complete review corpus

---

## Week 4 Checklist (to be completed)

- [ ] reports/Week_4_Report.docx created
- [ ] src/eda.py script created and runs without errors
- [ ] All charts saved to figures/ with descriptive filenames
- [ ] Findings clearly distinguished from planned analyses
- [ ] Skin-type comparisons only made if adequate sample sizes exist

---

## Week 5 Checklist (to be completed)

- [ ] reports/Week_5_Report.docx created
- [ ] src/model.py script created and runs without errors
- [ ] Baseline established before any model is trained
- [ ] At least two models compared
- [ ] Leakage-aware train/validation/test split used
- [ ] All metrics reported from executed experiments only
- [ ] Random seeds fixed

---

## Week 6 Checklist (to be completed)

- [ ] reports/Week_6_Report.docx created
- [ ] Results table populated only from executed experiments
- [ ] Privacy and fairness considerations addressed
- [ ] Limitations stated proportional to actual evidence
- [ ] Disclaimer that results do not constitute medical advice

---

## Final Verification Checklist (intern to complete before submission)

- [ ] Confirm dataset license is acceptable for academic use
- [ ] Confirm all model metrics shown in reports match what the scripts actually output
- [ ] Confirm all figures referenced in reports exist in the figures/ directory
- [ ] Confirm the GitHub repository URL is correct
- [ ] Confirm no raw data files are committed to a public repository (if license restricts)
- [ ] Read the executive summary of each report and check it accurately describes the work
