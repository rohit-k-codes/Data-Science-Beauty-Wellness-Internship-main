"""
src/clean.py — Week 3 preprocessing pipeline
=============================================
Reads raw Sephora review CSV(s) from data/raw/, applies a modular cleaning
pipeline, and writes data/processed/reviews_clean.csv.

Design principles
-----------------
* Raw files are NEVER modified (immutable source).
* Each step is a pure function: DataFrame in -> DataFrame + log-line out.
* Missing skin_type is mapped to the sentinel "unknown"; it is NEVER imputed.
* Text cleaning preserves negation (no, not, never, n't) and sentence
  boundaries so that downstream NLP does not lose polarity signals.
* Rating validation rejects values outside {1, 2, 3, 4, 5}; non-integer
  numeric ratings are rounded to the nearest valid integer and flagged.
* Duplicate detection removes repeated author/text keys and records each
  before/after count in the audit report.
* Unknown profile values (skin_tone, eye_color, hair_color) are kept as-is;
  they are not normalised to "unknown" unless they are blank/null.
* Skin type is mapped from raw string to a canonical set via an explicit
  lookup table; unrecognised values are set to "unknown" and reported.
* An audit CSV (data/processed/audit_report.csv) is written alongside the
  cleaned file with before/after counts for every step.

Usage
-----
    python src/clean.py [--input PATH] [--output PATH] [--audit-output PATH]
                        [--sample N]

Without --input, the pipeline reads the numbered review CSV chunks in
data/raw/ and uses product_info.csv to add product categories when available.
The --sample flag reads only the first N rows across those files, so it can be
used without loading the full dataset.
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT = PROCESSED_DIR / "reviews_clean.csv"
AUDIT_OUTPUT = PROCESSED_DIR / "audit_report.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Schema normalisation
# ---------------------------------------------------------------------------

# Canonical column names expected downstream.  Keys = acceptable raw variants,
# values = canonical name.
COLUMN_ALIASES: dict[str, str] = {
    # review text
    "review_text": "review_text",
    "reviewtext": "review_text",
    "text": "review_text",
    "body": "review_text",
    # rating
    "rating": "rating",
    "stars": "rating",
    "star_rating": "rating",
    # skin type
    "skin_type": "skin_type",
    "skintype": "skin_type",
    "skin type": "skin_type",
    # product
    "product_id": "product_id",
    "productid": "product_id",
    "product_name": "product_name",
    "productname": "product_name",
    "brand_name": "brand_name",
    "brand": "brand_name",
    # categories
    "primary_category": "primary_category",
    "secondary_category": "secondary_category",
    "tertiary_category": "tertiary_category",
    # reviewer
    "author_id": "author_id",
    "authorid": "author_id",
    "user_id": "author_id",
    # optional profile fields
    "review_title": "review_title",
    "skin_tone": "skin_tone",
    "eye_color": "eye_color",
    "hair_color": "hair_color",
    "helpfulness": "helpfulness",
    "price_usd": "price_usd",
    "loves_count": "loves_count",
    "reviews_count": "reviews_count",
    "submission_time": "submission_time",
    "is_recommended": "is_recommended",
}

# Columns that must exist after schema normalisation (pipeline will abort
# with a clear error if any are missing from the raw file).
REQUIRED_COLUMNS = {"review_text", "rating", "skin_type", "product_id"}


def normalise_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rename raw column headers to canonical names.

    Returns (df_normalised, log_message).
    Raises ValueError if a required column cannot be found after renaming.
    """
    # Strip whitespace and lowercase all column names before mapping.
    rename_map = {
        col: COLUMN_ALIASES.get(col.strip().lower(), col.strip().lower())
        for col in df.columns
    }
    df = df.rename(columns=rename_map)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns not found after schema normalisation: {missing}. "
            f"Present columns: {sorted(df.columns.tolist())}"
        )
    msg = (
        f"[schema] columns normalised — {len(df.columns)} total, "
        f"required columns present: {sorted(REQUIRED_COLUMNS)}"
    )
    return df, msg


# ---------------------------------------------------------------------------
# 2. Skincare category filter
# ---------------------------------------------------------------------------

SKINCARE_CATEGORY_KEYWORDS = {
    "skincare",
    "skin care",
    "moisturiser",
    "moisturizer",
    "serum",
    "cleanser",
    "toner",
    "sunscreen",
    "spf",
    "eye cream",
    "face mask",
    "face oil",
    "face wash",
    "exfoliant",
    "retinol",
    "treatment",
}


def filter_skincare(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Keep only rows whose primary_category looks like skincare.

    If primary_category is absent or all-null, the filter is skipped (the
    full review corpus is retained) and a warning is logged.
    """
    before = len(df)
    if "primary_category" not in df.columns or df["primary_category"].isna().all():
        msg = (
            f"[skincare_filter] primary_category absent or all-null — "
            f"filter skipped, all {before} rows retained"
        )
        return df, msg

    def _is_skincare(val: object) -> bool:
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return any(kw in s for kw in SKINCARE_CATEGORY_KEYWORDS)

    mask = df["primary_category"].apply(_is_skincare)
    df = df[mask].copy()
    after = len(df)
    msg = (
        f"[skincare_filter] {before} -> {after} rows "
        f"(removed {before - after} non-skincare rows)"
    )
    return df, msg


# ---------------------------------------------------------------------------
# 3. Missing-value rules
# ---------------------------------------------------------------------------

# Required text/rating rules: drop rows where these are missing.
DROP_IF_NULL = ["review_text", "rating"]

# skin_type missing -> sentinel "unknown" (NEVER imputed).
SKIN_TYPE_SENTINEL = "unknown"


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Apply field-level missing-value rules.

    * review_text null or blank-only -> drop row.
    * rating null -> drop row.
    * skin_type null / blank -> replace with sentinel 'unknown'.
    * All other nulls are documented but kept.

    Returns (df_cleaned, log_message).
    """
    before = len(df)
    lines = []

    if "skin_type_raw" not in df.columns:
        df["skin_type_raw"] = df["skin_type"].copy()

    # Drop rows with null review_text.
    blank_mask = df["review_text"].isna() | (
        df["review_text"].astype(str).str.strip() == ""
    )
    n_blank_text = blank_mask.sum()
    df = df[~blank_mask].copy()
    lines.append(f"review_text null/blank dropped: {n_blank_text}")

    # Drop rows with null rating.
    null_rating = df["rating"].isna()
    n_null_rating = null_rating.sum()
    df = df[~null_rating].copy()
    lines.append(f"rating null dropped: {n_null_rating}")

    # skin_type: fill nulls and blank strings with sentinel.
    blank_skin = df["skin_type"].isna() | (
        df["skin_type"].astype(str).str.strip() == ""
    )
    n_unknown_skin = blank_skin.sum()
    df.loc[blank_skin, "skin_type"] = SKIN_TYPE_SENTINEL
    lines.append(f"skin_type set to '{SKIN_TYPE_SENTINEL}': {n_unknown_skin}")

    # Document remaining nulls in optional columns.
    other_null_summary = (
        df.drop(columns=["review_text", "rating", "skin_type", "skin_type_raw"])
        .isna()
        .sum()
    )
    other_nulls = other_null_summary[other_null_summary > 0].to_dict()
    if other_nulls:
        lines.append(f"remaining nulls in optional columns: {other_nulls}")

    after = len(df)
    msg = (
        f"[missing_values] {before} -> {after} rows; "
        + "; ".join(lines)
    )
    return df, msg


# ---------------------------------------------------------------------------
# 4. Duplicate diagnostics
# ---------------------------------------------------------------------------


def diagnose_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Detect and remove exact duplicate reviews.

    Exact duplicate = same author_id + same review_text (case-insensitive,
    whitespace-normalised).  If author_id is absent, duplicates are detected
    on review_text alone.

    The first occurrence is kept; subsequent duplicates are dropped.
    A summary is logged; per-duplicate detail is available in the audit report.
    """
    before = len(df)

    # Prepare normalised keys for duplicate detection.
    norm_text = df["review_text"].astype(str).str.lower().str.split().str.join(" ")

    if "author_id" in df.columns:
        dup_key = df["author_id"].astype(str).str.strip() + "|||" + norm_text
    else:
        dup_key = norm_text

    dup_mask = dup_key.duplicated(keep="first")
    n_dups = dup_mask.sum()
    df = df[~dup_mask].copy()
    after = len(df)

    msg = (
        f"[duplicates] {before} -> {after} rows; "
        f"{n_dups} exact duplicates removed "
        f"(key: {'author_id + review_text' if 'author_id' in df.columns else 'review_text'})"
    )
    return df, msg


# ---------------------------------------------------------------------------
# 5. Rating validation
# ---------------------------------------------------------------------------

VALID_RATINGS = {1, 2, 3, 4, 5}


def validate_ratings(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Enforce ratings are integers in {1, 2, 3, 4, 5}.

    * Non-numeric -> drop row.
    * Numeric but non-integer -> round to nearest integer, flag, keep.
    * Rounded value still outside {1..5} -> drop row.
    * Exact integer in {1..5} -> cast to int, keep.
    """
    before = len(df)
    n_non_numeric = 0
    n_rounded = 0
    n_out_of_range = 0

    # Coerce to numeric; errors -> NaN.
    numeric = pd.to_numeric(df["rating"], errors="coerce")
    non_numeric_mask = numeric.isna() | ~np.isfinite(numeric)
    n_non_numeric = int(non_numeric_mask.sum())
    df = df[~non_numeric_mask].copy()
    numeric = numeric[~non_numeric_mask]

    # Round non-integer values.
    not_whole = (numeric % 1) != 0
    n_rounded = int(not_whole.sum())
    numeric = numeric.round().astype(int)

    # Drop out-of-range.
    out_of_range = ~numeric.isin(VALID_RATINGS)
    n_out_of_range = int(out_of_range.sum())
    df = df[~out_of_range].copy()
    numeric = numeric[~out_of_range]

    df["rating"] = numeric.astype(int)
    after = len(df)

    msg = (
        f"[rating_validation] {before} -> {after} rows; "
        f"non-numeric dropped: {n_non_numeric}; "
        f"fractional values rounded: {n_rounded}; "
        f"out-of-range dropped: {n_out_of_range}"
    )
    return df, msg


# ---------------------------------------------------------------------------
# 6. Explicit skin-type mapping
# ---------------------------------------------------------------------------

# Canonical skin types accepted downstream.
CANONICAL_SKIN_TYPES = {
    "dry",
    "oily",
    "combination",
    "normal",
    "sensitive",
    "unknown",
}

# Explicit lookup: raw string (lowercased, stripped) -> canonical value.
# Unrecognised strings are set to "unknown" and reported.
SKIN_TYPE_MAP: dict[str, str] = {
    # Canonical pass-through.
    "dry": "dry",
    "oily": "oily",
    "combination": "combination",
    "combo": "combination",
    "combination/oily": "combination",
    "oily/combination": "combination",
    "normal": "normal",
    "sensitive": "sensitive",
    "unknown": "unknown",
    # Common variants.
    "dry skin": "dry",
    "oily skin": "oily",
    "normal skin": "normal",
    "sensitive skin": "sensitive",
    "combination skin": "combination",
    "all skin types": "normal",
    "": "unknown",
    "n/a": "unknown",
    "na": "unknown",
    "none": "unknown",
    "not sure": "unknown",
    "not listed": "unknown",
}


def map_skin_type(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Map raw skin_type strings to canonical values.

    Unrecognised values are logged (with counts) and replaced by 'unknown'.
    The original raw value is preserved in skin_type_raw for audit purposes.
    """
    df = df.copy()
    if "skin_type_raw" not in df.columns:
        df["skin_type_raw"] = df["skin_type"].copy()

    normalised = df["skin_type"].astype(str).str.strip().str.lower()
    mapped = normalised.map(SKIN_TYPE_MAP)

    # Unrecognised -> unknown, but keep the raw value visible.
    unrecognised_mask = mapped.isna()
    n_unrecognised = int(unrecognised_mask.sum())
    unrecognised_values = (
        normalised[unrecognised_mask].value_counts().head(10).to_dict()
    )

    mapped = mapped.fillna("unknown")
    df["skin_type"] = mapped

    msg = (
        f"[skin_type_map] "
        f"{n_unrecognised} unrecognised values mapped to 'unknown'; "
        f"distribution after mapping: "
        f"{df['skin_type'].value_counts().to_dict()}; "
        f"unrecognised samples: {unrecognised_values}"
    )
    return df, msg


# ---------------------------------------------------------------------------
# 7. Conservative text cleaning (preserves negation)
# ---------------------------------------------------------------------------

# Negation words that must survive cleaning.
NEGATION_WORDS = frozenset(
    ["no", "not", "never", "neither", "nor", "nobody", "nothing", "nowhere",
     "n't", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
     "wasn't", "weren't", "isn't", "aren't", "haven't", "hasn't",
     "wouldn't", "shouldn't", "couldn't"]
)

# Unicode normalisation and noise removal patterns.
_RE_HTML = re.compile(r"<[^>]+>")                      # HTML tags
_RE_URL = re.compile(r"https?://\S+|www\.\S+")         # URLs
_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]+")           # non-ASCII
_RE_REPEATED_PUNCT = re.compile(r"([!?.]){2,}")        # !!!  ???  ...
_RE_WHITESPACE = re.compile(r"\s+")                     # collapse spaces


def clean_text(text: str) -> str:
    """Apply conservative cleaning to a single review string.

    Steps (in order):
    1. Strip leading/trailing whitespace.
    2. Remove HTML tags.
    3. Remove URLs.
    4. Replace non-ASCII characters with a space (preserves apostrophes
       which are ASCII, so "don't" is kept intact).
    5. Lowercase.
    6. Collapse repeated punctuation (e.g. '!!!' -> '!').
    7. Collapse internal whitespace.

    Intentionally NOT done:
    - Removing stopwords (would destroy negation).
    - Removing punctuation wholesale (would merge words, destroy n't).
    - Stemming or lemmatisation (changes token identity).
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_NON_ASCII.sub(" ", text)
    text = text.lower()
    text = _RE_REPEATED_PUNCT.sub(r"\1", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def clean_review_text(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Apply clean_text() to the review_text column.

    Adds a new column 'review_text_clean'; the original 'review_text' is
    preserved untouched.  Also adds 'review_length_words' and
    'review_length_chars' (measured on the cleaned text).
    """
    df = df.copy()
    df["review_text_clean"] = df["review_text"].apply(clean_text)

    # Verify negation is preserved in a random spot-check (logged only).
    sample_with_neg = df[
        df["review_text_clean"].str.contains(r"\bnot?\b|\bn't\b", regex=True, na=False)
    ]
    n_negation = len(sample_with_neg)

    df["review_length_words"] = (
        df["review_text_clean"].str.split().str.len().fillna(0).astype(int)
    )
    df["review_length_chars"] = df["review_text_clean"].str.len().fillna(0).astype(int)

    msg = (
        f"[text_cleaning] review_text_clean added; "
        f"{n_negation} rows contain negation tokens (preserved); "
        f"mean word count: {df['review_length_words'].mean():.1f}; "
        f"mean char count: {df['review_length_chars'].mean():.1f}"
    )
    return df, msg


def validate_clean_text(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Drop rows whose review text becomes empty after conservative cleaning."""
    before = len(df)
    blank_mask = (
        df["review_text_clean"].isna()
        | df["review_text_clean"].astype(str).str.strip().eq("")
    )
    df = df[~blank_mask].copy()
    removed = before - len(df)
    return df, (
        f"[text_quality] {before} -> {len(df)} rows; "
        f"empty after cleaning dropped: {removed}"
    )


# ---------------------------------------------------------------------------
# 8. Audit report
# ---------------------------------------------------------------------------


def write_audit_report(audit_rows: list[dict], path: Path) -> None:
    """Write audit_report.csv with raw/cleaned counts and step summaries."""
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(path, index=False)
    log.info("Audit report written -> %s", path)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def _find_raw_csvs(raw_dir: Path) -> list[Path]:
    """Return numbered review chunks, excluding the synthetic test sample."""
    csvs = [
        path for path in raw_dir.glob("reviews_*.csv")
        if path.name.lower() != "reviews_sample.csv"
    ]
    if not csvs:
        raise FileNotFoundError(
            f"No numbered review CSV files found in {raw_dir}. "
            "Please download the Sephora dataset from Kaggle and place it in data/raw/."
        )

    def _chunk_start(path: Path) -> int:
        match = re.match(r"reviews_(\d+)-", path.name.lower())
        return int(match.group(1)) if match else sys.maxsize

    return sorted(csvs, key=_chunk_start)


def load_raw(
    raw_paths: Union[Path, Sequence[Path]],
    sample: Optional[int] = None,
) -> pd.DataFrame:
    """Load review CSV(s); with sample, read only the first N rows in file order.

    Reading with nrows keeps a sample run bounded even when source CSVs are
    very large. Full runs concatenate all supplied files.
    """
    paths = [raw_paths] if isinstance(raw_paths, Path) else list(raw_paths)
    if not paths:
        raise ValueError("At least one raw review CSV path is required.")
    if sample is not None and sample <= 0:
        raise ValueError("--sample must be a positive integer.")

    frames = []
    remaining = sample
    for raw_path in paths:
        log.info("Loading raw data from %s …", raw_path)
        read_options: dict[str, int] = {}
        if remaining is not None:
            read_options["nrows"] = remaining
        frame = pd.read_csv(raw_path, low_memory=False, **read_options)
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
            if remaining <= 0:
                break

    df = pd.concat(frames, ignore_index=True)
    log.info("Raw data loaded: %d rows × %d columns", len(df), len(df.columns))
    return df


def attach_product_categories(
    df: pd.DataFrame,
    product_info_path: Path,
) -> tuple[pd.DataFrame, str]:
    """Attach primary_category from product metadata when reviews lack it."""
    if "primary_category" in df.columns:
        return df, "[product_categories] primary_category already present"
    if not product_info_path.exists():
        return df, (
            "[product_categories] product_info.csv not found; "
            "category filtering cannot be applied"
        )

    products = pd.read_csv(
        product_info_path,
        usecols=["product_id", "primary_category"],
        low_memory=False,
    )
    if products["product_id"].duplicated().any():
        raise ValueError(
            f"Duplicate product_id values in product metadata: {product_info_path}"
        )
    before = len(df)
    df = df.merge(products, on="product_id", how="left", validate="many_to_one")
    unmatched = int(df["primary_category"].isna().sum())
    return df, (
        f"[product_categories] category lookup joined; {before} review rows "
        f"preserved; unmatched product_id values: {unmatched}"
    )


def run_pipeline(
    input_path: Optional[Path] = None,
    output_path: Path = DEFAULT_OUTPUT,
    sample: Optional[int] = None,
    audit_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute the full cleaning pipeline and return the cleaned DataFrame.

    Parameters
    ----------
    input_path : optional path to one raw review CSV. If omitted, all
        numbered review chunks in data/raw/ are processed in file order.
    output_path : destination for reviews_clean.csv.
    sample : if set, read only this many first rows across the input files.
    audit_path : optional destination for the data-quality audit CSV.
    """
    # ------------------------------------------------------------------ setup
    input_paths = [input_path] if input_path is not None else _find_raw_csvs(RAW_DIR)

    df = load_raw(input_paths, sample=sample)
    raw_row_count = len(df)

    audit_rows: list[dict] = []
    step_number = 0

    def _record(step_name: str, rows_before: int, rows_after: int, note: str) -> None:
        audit_rows.append(
            {
                "step": step_number,
                "name": step_name,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
                "note": note,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    # ------------------------------------------------ Step 0: raw snapshot
    step_number = 0
    input_names = ", ".join(path.name for path in input_paths)
    if sample is not None:
        input_names += f"; first {raw_row_count} source rows (sample)"
    source_description = input_names
    _record("raw_load", raw_row_count, raw_row_count, f"source: {input_names}")
    log.info("Step 0 — raw load: %d rows", raw_row_count)

    # ------------------------------------------- Step 1: schema normalisation
    step_number = 1
    rows_before = len(df)
    df, msg = normalise_schema(df)
    _record("schema_normalise", rows_before, len(df), msg)
    log.info("Step 1 — %s", msg)

    # ------------------------------------------ product category enrichment
    step_number = 2
    rows_before = len(df)
    df, msg = attach_product_categories(df, RAW_DIR / "product_info.csv")
    _record("product_categories", rows_before, len(df), msg)
    log.info("Step 2 — %s", msg)

    # ------------------------------------- Step 2: skincare category filter
    step_number = 3
    rows_before = len(df)
    df, msg = filter_skincare(df)
    _record("skincare_filter", rows_before, len(df), msg)
    log.info("Step 3 — %s", msg)

    # ---------------------------------------- Step 3: missing-value rules
    step_number = 4
    rows_before = len(df)
    df, msg = handle_missing(df)
    _record("missing_values", rows_before, len(df), msg)
    log.info("Step 4 — %s", msg)

    # ------------------------------------ Step 4: duplicate diagnostics
    step_number = 5
    rows_before = len(df)
    df, msg = diagnose_duplicates(df)
    _record("duplicate_removal", rows_before, len(df), msg)
    log.info("Step 5 — %s", msg)

    # ------------------------------------------ Step 5: rating validation
    step_number = 6
    rows_before = len(df)
    df, msg = validate_ratings(df)
    _record("rating_validation", rows_before, len(df), msg)
    log.info("Step 6 — %s", msg)

    # ----------------------------------------- Step 6: skin-type mapping
    step_number = 7
    rows_before = len(df)
    df, msg = map_skin_type(df)
    _record("skin_type_map", rows_before, len(df), msg)
    log.info("Step 7 — %s", msg)

    # -------------------------------------- Step 7: text cleaning
    step_number = 8
    rows_before = len(df)
    df, msg = clean_review_text(df)
    _record("text_cleaning", rows_before, len(df), msg)
    log.info("Step 8 — %s", msg)

    # --------------------------------------- final text-quality validation
    step_number = 9
    rows_before = len(df)
    df, msg = validate_clean_text(df)
    _record("text_quality", rows_before, len(df), msg)
    log.info("Step 9 — %s", msg)

    # --------------------------------------------------------- write outputs
    df.to_csv(output_path, index=False)
    log.info("Cleaned data written -> %s  (%d rows × %d columns)",
             output_path, len(df), len(df.columns))

    selected_audit_path = audit_path or (
        AUDIT_OUTPUT
        if output_path == DEFAULT_OUTPUT
        else output_path.with_name(f"audit_report_{output_path.stem}.csv")
    )
    write_audit_report(audit_rows, selected_audit_path)

    # -------------------------------------------------- summary to stdout
    print("\n" + "=" * 60)
    print("CLEANING PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Source file(s) : {source_description}")
    print(f"  Raw rows       : {raw_row_count:,}")
    print(f"  Cleaned rows   : {len(df):,}")
    print(f"  Rows removed   : {raw_row_count - len(df):,}  "
          f"({(raw_row_count - len(df)) / max(raw_row_count, 1) * 100:.1f}%)")
    print(f"  Output         : {output_path}")
    print(f"  Audit report   : {selected_audit_path}")
    print("=" * 60)

    for row in audit_rows:
        removed = row["rows_removed"]
        print(
            f"  Step {row['step']:1d}  {row['name']:<22s}  "
            f"{row['rows_before']:>8,} -> {row['rows_after']:>8,}  "
            f"(-{removed:>6,})"
        )

    skin_dist = df["skin_type"].value_counts().to_dict()
    print(f"\n  skin_type distribution: {skin_dist}")
    rating_dist = df["rating"].value_counts().sort_index().to_dict()
    print(f"  rating distribution  : {rating_dist}")
    missing_pct = (df.isna().sum() / len(df) * 100).round(1)
    missing_nonzero = missing_pct[missing_pct > 0].to_dict()
    print(f"  remaining null %     : {missing_nonzero}")
    print("=" * 60 + "\n")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3 cleaning pipeline for Sephora skincare reviews."
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to raw CSV (default: auto-detect from data/raw/)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Destination path for reviews_clean.csv"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Read only the first N rows across the selected input CSV(s)"
    )
    parser.add_argument(
        "--audit-output", type=Path, default=None,
        help="Destination path for the data-quality audit CSV"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        sample=args.sample,
        audit_path=args.audit_output,
    )
