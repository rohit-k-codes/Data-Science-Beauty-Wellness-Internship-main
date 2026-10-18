"""
tests/test_clean.py — edge-case tests for src/clean.py
========================================================
Run with: python -m pytest tests/ -v

Each test exercises a single pipeline step with a hand-crafted DataFrame so
that no raw data file is required.  Tests are fast, deterministic, and
independent.
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure src/ is importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clean import (
    normalise_schema,
    filter_skincare,
    handle_missing,
    diagnose_duplicates,
    validate_ratings,
    map_skin_type,
    clean_text,
    clean_review_text,
    validate_clean_text,
    load_raw,
    attach_product_categories,
    _find_raw_csvs,
    SKIN_TYPE_SENTINEL,
    CANONICAL_SKIN_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_df(**kwargs) -> pd.DataFrame:
    """Return a minimal valid DataFrame with required columns.

    Keyword arguments override defaults.
    """
    defaults = {
        "review_text": ["Great moisturiser, love it!"],
        "rating": [5],
        "skin_type": ["dry"],
        "product_id": ["P001"],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ===========================================================================
# 1. Schema normalisation
# ===========================================================================

class TestNormaliseSchema:
    def test_canonical_columns_unchanged(self):
        df = _minimal_df()
        result, msg = normalise_schema(df)
        assert "review_text" in result.columns
        assert "rating" in result.columns

    def test_alias_renamed(self):
        df = pd.DataFrame(
            {"reviewtext": ["text"], "stars": [4], "skintype": ["oily"],
             "productid": ["P1"]}
        )
        result, _ = normalise_schema(df)
        assert "review_text" in result.columns
        assert "rating" in result.columns
        assert "skin_type" in result.columns
        assert "product_id" in result.columns

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"review_text": ["text"], "rating": [5]})
        # skin_type and product_id are missing
        with pytest.raises(ValueError, match="Required columns not found"):
            normalise_schema(df)

    def test_extra_columns_preserved(self):
        df = _minimal_df(brand_name=["Neutrogena"], price_usd=[25.0])
        result, _ = normalise_schema(df)
        assert "brand_name" in result.columns
        assert "price_usd" in result.columns

    def test_whitespace_stripped_from_column_names(self):
        df = pd.DataFrame(
            {" review_text ": ["text"], " rating ": [3],
             " skin_type ": ["dry"], " product_id ": ["P1"]}
        )
        result, _ = normalise_schema(df)
        assert "review_text" in result.columns


# ===========================================================================
# 2. Skincare category filter
# ===========================================================================

class TestFilterSkincare:
    def test_keeps_skincare_rows(self):
        df = pd.DataFrame(
            {"review_text": ["a", "b", "c"],
             "rating": [5, 4, 3],
             "skin_type": ["dry", "oily", "normal"],
             "product_id": ["P1", "P2", "P3"],
             "primary_category": ["Skincare", "Makeup", "Skincare"]}
        )
        result, msg = filter_skincare(df)
        assert len(result) == 2
        assert "[skincare_filter]" in msg

    def test_absent_category_skips_filter(self):
        df = _minimal_df()  # no primary_category column
        result, msg = filter_skincare(df)
        assert len(result) == 1  # all rows kept
        assert "filter skipped" in msg

    def test_all_null_category_skips_filter(self):
        df = _minimal_df(primary_category=[None])
        result, msg = filter_skincare(df)
        assert len(result) == 1
        assert "filter skipped" in msg

    def test_case_insensitive_match(self):
        df = pd.DataFrame(
            {"review_text": ["a"],
             "rating": [5],
             "skin_type": ["dry"],
             "product_id": ["P1"],
             "primary_category": ["SKINCARE"]}
        )
        result, _ = filter_skincare(df)
        assert len(result) == 1


# ===========================================================================
# 3. Missing-value rules
# ===========================================================================

class TestHandleMissing:
    def test_null_review_text_dropped(self):
        df = pd.DataFrame(
            {"review_text": [None, "good", "  "],
             "rating": [5, 4, 3],
             "skin_type": ["dry", "oily", "normal"],
             "product_id": ["P1", "P2", "P3"]}
        )
        result, msg = handle_missing(df)
        assert len(result) == 1  # only "good" survives
        assert result["review_text"].iloc[0] == "good"

    def test_blank_review_text_dropped(self):
        df = _minimal_df(review_text=["   "])
        result, _ = handle_missing(df)
        assert len(result) == 0

    def test_null_rating_dropped(self):
        df = pd.DataFrame(
            {"review_text": ["a", "b"],
             "rating": [None, 4],
             "skin_type": ["dry", "oily"],
             "product_id": ["P1", "P2"]}
        )
        result, _ = handle_missing(df)
        assert len(result) == 1
        assert result["rating"].iloc[0] == 4

    def test_null_skin_type_becomes_unknown(self):
        df = pd.DataFrame(
            {"review_text": ["a", "b", "c"],
             "rating": [5, 4, 3],
             "skin_type": [None, "", "dry"],
             "product_id": ["P1", "P2", "P3"]}
        )
        result, _ = handle_missing(df)
        assert result["skin_type"].tolist() == ["unknown", "unknown", "dry"]

    def test_original_skin_type_value_is_preserved(self):
        df = _minimal_df(skin_type=[None])
        result, _ = handle_missing(df)
        result, _ = map_skin_type(result)

        assert result["skin_type"].tolist() == ["unknown"]
        assert pd.isna(result["skin_type_raw"].iloc[0])

    def test_skin_type_unknown_not_dropped(self):
        """Missing skin_type must NOT cause row deletion."""
        df = pd.DataFrame(
            {"review_text": ["text"],
             "rating": [3],
             "skin_type": [None],
             "product_id": ["P1"]}
        )
        result, _ = handle_missing(df)
        assert len(result) == 1

    def test_optional_nulls_retained(self):
        """Nulls in optional columns (brand_name) should remain in data."""
        df = pd.DataFrame(
            {"review_text": ["text"],
             "rating": [3],
             "skin_type": ["dry"],
             "product_id": ["P1"],
             "brand_name": [None]}
        )
        result, _ = handle_missing(df)
        assert len(result) == 1
        assert pd.isna(result["brand_name"].iloc[0])


# ===========================================================================
# 4. Duplicate diagnostics
# ===========================================================================

class TestDiagnoseDuplicates:
    def test_exact_duplicate_removed(self):
        df = pd.DataFrame(
            {"review_text": ["same text", "same text"],
             "rating": [5, 5],
             "skin_type": ["dry", "dry"],
             "product_id": ["P1", "P1"],
             "author_id": ["U1", "U1"]}
        )
        result, msg = diagnose_duplicates(df)
        assert len(result) == 1
        assert "1 exact duplicates removed" in msg

    def test_different_authors_not_duplicate(self):
        df = pd.DataFrame(
            {"review_text": ["same text", "same text"],
             "rating": [5, 4],
             "skin_type": ["dry", "oily"],
             "product_id": ["P1", "P2"],
             "author_id": ["U1", "U2"]}
        )
        result, _ = diagnose_duplicates(df)
        assert len(result) == 2

    def test_case_insensitive_dedup(self):
        df = pd.DataFrame(
            {"review_text": ["Great product!", "great product!"],
             "rating": [5, 5],
             "skin_type": ["dry", "dry"],
             "product_id": ["P1", "P1"],
             "author_id": ["U1", "U1"]}
        )
        result, _ = diagnose_duplicates(df)
        assert len(result) == 1

    def test_whitespace_normalised_for_dedup(self):
        df = pd.DataFrame(
            {"review_text": ["Great   product", "Great product"],
             "rating": [5, 5],
             "skin_type": ["dry", "dry"],
             "product_id": ["P1", "P1"],
             "author_id": ["U1", "U1"]}
        )
        result, _ = diagnose_duplicates(df)
        assert len(result) == 1

    def test_no_author_id_dedup_by_text(self):
        df = pd.DataFrame(
            {"review_text": ["copy", "copy", "unique"],
             "rating": [5, 5, 4],
             "skin_type": ["dry", "dry", "oily"],
             "product_id": ["P1", "P1", "P2"]}
        )
        result, _ = diagnose_duplicates(df)
        assert len(result) == 2

    def test_first_occurrence_kept(self):
        df = pd.DataFrame(
            {"review_text": ["text", "text"],
             "rating": [5, 3],
             "skin_type": ["dry", "oily"],
             "product_id": ["P1", "P1"],
             "author_id": ["U1", "U1"]}
        )
        result, _ = diagnose_duplicates(df)
        assert result["rating"].iloc[0] == 5


# ===========================================================================
# 5. Rating validation
# ===========================================================================

class TestValidateRatings:
    def test_valid_integers_kept(self):
        df = pd.DataFrame(
            {"review_text": list("abcde"),
             "rating": [1, 2, 3, 4, 5],
             "skin_type": ["dry"] * 5,
             "product_id": [f"P{i}" for i in range(5)]}
        )
        result, msg = validate_ratings(df)
        assert len(result) == 5
        assert list(result["rating"]) == [1, 2, 3, 4, 5]

    def test_non_numeric_dropped(self):
        df = pd.DataFrame(
            {"review_text": ["a", "b"],
             "rating": ["five", 4],
             "skin_type": ["dry", "oily"],
             "product_id": ["P1", "P2"]}
        )
        result, msg = validate_ratings(df)
        assert len(result) == 1
        assert "non-numeric dropped: 1" in msg

    def test_out_of_range_dropped(self):
        df = pd.DataFrame(
            {"review_text": ["a", "b", "c"],
             "rating": [0, 3, 6],
             "skin_type": ["dry", "oily", "normal"],
             "product_id": ["P1", "P2", "P3"]}
        )
        result, _ = validate_ratings(df)
        assert len(result) == 1
        assert result["rating"].iloc[0] == 3

    def test_fractional_rounded(self):
        df = pd.DataFrame(
            {"review_text": ["a", "b"],
             "rating": [4.6, 1.4],
             "skin_type": ["dry", "oily"],
             "product_id": ["P1", "P2"]}
        )
        result, msg = validate_ratings(df)
        # 4.6 → 5, 1.4 → 1 — both valid
        assert set(result["rating"].tolist()) == {5, 1}
        assert "fractional values rounded: 2" in msg

    def test_fractional_rounds_to_valid_range(self):
        df = pd.DataFrame(
            {"review_text": ["a"],
             "rating": [5.4],
             "skin_type": ["dry"],
             "product_id": ["P1"]}
        )
        result, _ = validate_ratings(df)
        assert result["rating"].iloc[0] == 5

    def test_rating_dtype_is_int(self):
        df = _minimal_df(rating=[4])
        result, _ = validate_ratings(df)
        assert result["rating"].dtype == int

    def test_infinite_rating_is_dropped(self):
        df = pd.DataFrame(
            {"review_text": ["bad rating", "valid rating"],
             "rating": [float("inf"), 4],
             "skin_type": ["dry", "oily"],
             "product_id": ["P1", "P2"]}
        )
        result, msg = validate_ratings(df)
        assert result["rating"].tolist() == [4]
        assert "non-numeric dropped: 1" in msg


# ===========================================================================
# 6. Skin-type mapping
# ===========================================================================

class TestMapSkinType:
    def test_canonical_values_unchanged(self):
        for val in ["dry", "oily", "combination", "normal", "sensitive"]:
            df = _minimal_df(skin_type=[val])
            result, _ = map_skin_type(df)
            assert result["skin_type"].iloc[0] == val

    def test_combo_mapped_to_combination(self):
        df = _minimal_df(skin_type=["combo"])
        result, _ = map_skin_type(df)
        assert result["skin_type"].iloc[0] == "combination"

    def test_unrecognised_mapped_to_unknown(self):
        df = _minimal_df(skin_type=["superoily123"])
        result, msg = map_skin_type(df)
        assert result["skin_type"].iloc[0] == "unknown"
        assert "unrecognised" in msg

    def test_raw_value_preserved(self):
        df = _minimal_df(skin_type=["combo"])
        result, _ = map_skin_type(df)
        assert "skin_type_raw" in result.columns
        assert result["skin_type_raw"].iloc[0] == "combo"

    def test_case_insensitive_mapping(self):
        df = _minimal_df(skin_type=["DRY"])
        result, _ = map_skin_type(df)
        assert result["skin_type"].iloc[0] == "dry"

    def test_all_output_values_canonical(self):
        df = pd.DataFrame(
            {"review_text": ["a"] * 7,
             "rating": [5] * 7,
             "skin_type": ["Dry", "OILY", "combo", "Normal", "Sensitive",
                           None, "weirdvalue"],
             "product_id": [f"P{i}" for i in range(7)]}
        )
        # Fill None before mapping (normally done by handle_missing first)
        df["skin_type"] = df["skin_type"].fillna("unknown")
        result, _ = map_skin_type(df)
        for val in result["skin_type"]:
            assert val in CANONICAL_SKIN_TYPES, f"Unexpected value: {val}"

    def test_empty_string_becomes_unknown(self):
        df = _minimal_df(skin_type=[""])
        result, _ = map_skin_type(df)
        assert result["skin_type"].iloc[0] == "unknown"


# ===========================================================================
# 7. Text cleaning — negation preservation
# ===========================================================================

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("This IS Great") == "this is great"

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_html_removed(self):
        assert "<b>" not in clean_text("This is <b>bold</b>!")

    def test_url_removed(self):
        result = clean_text("Check https://example.com for details")
        assert "https" not in result

    def test_negation_no_preserved(self):
        result = clean_text("This product is no good")
        assert "no" in result.split()

    def test_negation_not_preserved(self):
        result = clean_text("I do not recommend this")
        assert "not" in result.split()

    def test_negation_never_preserved(self):
        result = clean_text("I would never buy this again")
        assert "never" in result.split()

    def test_negation_contraction_preserved(self):
        result = clean_text("It doesn't work for me")
        assert "doesn't" in result or "doesn" in result

    def test_negation_cant_preserved(self):
        result = clean_text("I can't recommend this product")
        assert "can't" in result or "cant" in result or "not" in result or "n't" in result

    def test_repeated_punctuation_collapsed(self):
        result = clean_text("Amazing!!!")
        assert "!!!" not in result

    def test_non_ascii_replaced(self):
        result = clean_text("Café crème")
        # Non-ASCII chars removed but words survive
        assert "caf" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   ") == ""

    def test_none_coerced(self):
        # None input should not raise
        result = clean_text(None)
        assert isinstance(result, str)


class TestCleanReviewText:
    def test_adds_clean_column(self):
        df = _minimal_df(review_text=["Great product!"])
        result, _ = clean_review_text(df)
        assert "review_text_clean" in result.columns

    def test_original_preserved(self):
        df = _minimal_df(review_text=["ORIGINAL TEXT"])
        result, _ = clean_review_text(df)
        assert result["review_text"].iloc[0] == "ORIGINAL TEXT"

    def test_adds_length_columns(self):
        df = _minimal_df(review_text=["three word review"])
        result, _ = clean_review_text(df)
        assert "review_length_words" in result.columns
        assert "review_length_chars" in result.columns
        assert result["review_length_words"].iloc[0] == 3

    def test_negation_reported_in_log(self):
        df = pd.DataFrame(
            {"review_text": ["I do not like this", "I love this"],
             "rating": [1, 5],
             "skin_type": ["dry", "oily"],
             "product_id": ["P1", "P2"]}
        )
        _, msg = clean_review_text(df)
        # At least 1 row has negation
        assert "negation tokens" in msg


class TestTextQuality:
    def test_drops_text_empty_after_cleaning(self):
        df = pd.DataFrame(
            {"review_text": ["one", "two", "three"],
             "rating": [5, 4, 3],
             "skin_type": ["dry", "oily", "normal"],
             "product_id": ["P1", "P2", "P3"],
             "review_text_clean": ["", "   ", "usable review"]}
        )
        result, msg = validate_clean_text(df)
        assert result["review_text_clean"].tolist() == ["usable review"]
        assert "empty after cleaning dropped: 2" in msg


class TestRawLoading:
    def test_sample_reads_prefix_across_input_files(self, tmp_path):
        first = tmp_path / "reviews_0-2.csv"
        second = tmp_path / "reviews_2-end.csv"
        first.write_text("review_text,rating\nfirst,5\nsecond,4\n", encoding="utf-8")
        second.write_text("review_text,rating\nthird,3\n", encoding="utf-8")

        result = load_raw([first, second], sample=3)

        assert result["review_text"].tolist() == ["first", "second", "third"]

    def test_non_positive_sample_is_rejected(self, tmp_path):
        source = tmp_path / "reviews.csv"
        source.write_text("review_text,rating\nreview,5\n", encoding="utf-8")

        with pytest.raises(ValueError, match="positive integer"):
            load_raw(source, sample=0)

    def test_default_sources_are_numbered_review_chunks_only(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        for name in (
            "reviews_250-500.csv",
            "reviews_0-250.csv",
            "reviews_sample.csv",
            "product_info.csv",
        ):
            (raw_dir / name).touch()

        assert [path.name for path in _find_raw_csvs(raw_dir)] == [
            "reviews_0-250.csv",
            "reviews_250-500.csv",
        ]


class TestProductCategoryEnrichment:
    def test_adds_categories_by_product_id(self, tmp_path):
        products = tmp_path / "product_info.csv"
        products.write_text(
            "product_id,primary_category\nP1,Skincare\n",
            encoding="utf-8",
        )
        reviews = pd.DataFrame({"product_id": ["P1"], "review_text": ["text"]})

        result, msg = attach_product_categories(reviews, products)

        assert result["primary_category"].tolist() == ["Skincare"]
        assert "unmatched product_id values: 0" in msg


# ===========================================================================
# 8. Pipeline integration edge cases
# ===========================================================================

class TestPipelineIntegration:
    """Integration-level tests that pass a DataFrame through multiple steps."""

    def test_unknown_skin_type_survives_full_text_path(self):
        """A review with null skin_type must survive all steps and end as 'unknown'."""
        df = pd.DataFrame(
            {"review_text": ["Love this product! It does not cause breakouts."],
             "rating": [5],
             "skin_type": [None],
             "product_id": ["P001"]}
        )
        df, _ = normalise_schema(df)
        df, _ = handle_missing(df)
        df, _ = validate_ratings(df)
        df, _ = map_skin_type(df)
        df, _ = clean_review_text(df)
        assert df["skin_type"].iloc[0] == "unknown"
        assert len(df) == 1

    def test_negation_survives_full_pipeline(self):
        """Negation tokens in review_text must be present in review_text_clean."""
        text = "This moisturiser is not greasy and does not clog pores. Never going back!"
        df = pd.DataFrame(
            {"review_text": [text],
             "rating": [5],
             "skin_type": ["dry"],
             "product_id": ["P001"]}
        )
        df, _ = normalise_schema(df)
        df, _ = handle_missing(df)
        df, _ = validate_ratings(df)
        df, _ = map_skin_type(df)
        df, _ = clean_review_text(df)
        cleaned = df["review_text_clean"].iloc[0]
        assert "not" in cleaned.split()
        assert "never" in cleaned.split()

    def test_all_bad_rows_removed_gracefully(self):
        """A DataFrame where all rows are invalid should result in 0-row output without error."""
        df = pd.DataFrame(
            {"review_text": [None, "   ", ""],
             "rating": [5, 4, 3],
             "skin_type": ["dry", "oily", "normal"],
             "product_id": ["P1", "P2", "P3"]}
        )
        df, _ = normalise_schema(df)
        df, _ = handle_missing(df)
        assert len(df) == 0

    def test_fractional_rating_in_range_kept(self):
        """A rating of 4.8 should round to 5 and be kept."""
        df = _minimal_df(rating=[4.8])
        df, _ = normalise_schema(df)
        df, _ = validate_ratings(df)
        assert len(df) == 1
        assert df["rating"].iloc[0] == 5

    def test_unknown_skin_type_not_inferred(self):
        """Pipeline must NEVER change 'unknown' to any other value."""
        df = pd.DataFrame(
            {"review_text": ["my skin is very oily"],
             "rating": [3],
             "skin_type": [None],
             "product_id": ["P1"]}
        )
        df, _ = normalise_schema(df)
        df, _ = handle_missing(df)
        df, _ = map_skin_type(df)
        # Even though review text mentions oily, skin_type must remain "unknown"
        assert df["skin_type"].iloc[0] == "unknown"
