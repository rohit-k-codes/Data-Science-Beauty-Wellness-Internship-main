# Data Dictionary

**Project:** Analyzing Customer Reviews for Personalized Skincare Recommendations  
**Dataset:** Sephora Products and Skincare Reviews (Kaggle)  
**URL:** https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

---

## Download Record (to be filled in by intern)

| Field | Value |
|-------|-------|
| Download date | [FILL IN] |
| Exact license text from Kaggle page | [FILL IN] |
| File names downloaded | [FILL IN] |
| Total rows in raw file | [FILL IN] |
| Total columns in raw file | [FILL IN] |

---

## Field Definitions

| Field Name | Data Type | Values / Units | Required (R) or Optional (O) | Purpose in Analysis |
|------------|-----------|----------------|-------------------------------|---------------------|
| review_text | String | Free text, English | R | NLP input: keyword extraction and sentiment (RQ1) |
| rating | Integer | 1, 2, 3, 4, or 5 | R | Primary target variable; baseline recommendation signal (RQ2) |
| skin_type | Categorical string | dry, oily, combination, normal, sensitive (self-reported; may be blank) | R | Grouping variable for skin-type rating analysis (RQ2) |
| product_id | String | Alphanumeric identifier | R | Join key to product metadata; group-split identifier |
| product_name | String | Free text | R | Human-readable product label |
| brand_name | String | Free text | O | Brand-level grouping |
| primary_category | Categorical string | e.g. Skincare, Makeup, Fragrance | R | Filter to skincare subset; modelling feature |
| secondary_category | Categorical string | e.g. Moisturisers, Serums | O | More granular product grouping |
| tertiary_category | Categorical string | e.g. Eye Cream | O | Most granular category level |
| author_id | String | Anonymised user ID | O | Detect duplicate reviews from same user; group-based split |
| review_title | String | Free text | O | Additional NLP signal |
| skin_tone | Categorical string | e.g. fair, medium, deep (self-reported) | O | Secondary demographic variable; fairness analysis |
| eye_color | Categorical string | e.g. blue, brown, green | O | Demographic variable; not used in core model |
| hair_color | Categorical string | e.g. blonde, brunette | O | Demographic variable; not used in core model |
| helpfulness | Numeric or ratio | Format to be confirmed after download | O | Weight reviews; filter low-quality reviews |
| price_usd | Float | USD, e.g. 45.00 | O | Price-tier feature |
| loves_count | Integer | Count of product likes | O | Product popularity indicator |
| reviews_count | Integer | Total reviews per product | O | Product coverage indicator |
| submission_time | Datetime string | ISO date or Unix timestamp; format to be confirmed | O | Temporal trend analysis; recency weighting |
| is_recommended | Boolean or integer | 0 or 1 (if present in dataset) | O | Alternative binary classification target |

---

## Notes on Missing Values

- `skin_type` missing values: treated as **unknown**. Never imputed. Rows with missing skin_type are excluded from skin-type comparisons but retained for other analyses.
- `review_text` missing values: row is dropped (no review text = unusable for NLP).
- `rating` missing values: row is dropped (no target = unusable for modelling).
- All other missing values: documented in cleaning log; handled on a field-by-field basis.

---

## Engineered Features (added during Week 4)

| Feature Name | Source Field(s) | Type | Description |
|-------------|-----------------|------|-------------|
| review_length_words | review_text | Integer | Word count of cleaned review text |
| review_length_chars | review_text | Integer | Character count of cleaned review text |
| review_text_clean | review_text | String | Lowercased, stripped, punctuation removed |
| skin_type_known | skin_type | Boolean (0/1) | 1 if skin_type is not missing, else 0 |
| skin_type_encoded | skin_type | Integer | Label-encoded skin type (known values only) |
| kw_oily | review_text_clean | Boolean (0/1) | 1 if review contains oily skin keywords |
| kw_dry | review_text_clean | Boolean (0/1) | 1 if review contains dry skin keywords |
| kw_sensitive | review_text_clean | Boolean (0/1) | 1 if review contains sensitive skin keywords |
| kw_breakout | review_text_clean | Boolean (0/1) | 1 if review contains breakout/acne keywords |
| kw_hydration | review_text_clean | Boolean (0/1) | 1 if review contains hydration keywords |
| category_encoded | primary_category | Integer | Label-encoded primary category |
| high_rating | rating | Boolean (0/1) | 1 if rating >= 4, else 0 (classification target) |
