"""Tests for Week 4 descriptive feature engineering and EDA summaries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eda import build_summary, create_charts, engineer_features, load_reviews


def test_engineer_features_adds_descriptive_columns_without_mutating_input():
    source = pd.DataFrame(
        {
            "review_text_clean": ["good product", "not great", ""],
            "rating": [5, 2, 3],
            "skin_type": ["dry", "unknown", None],
            "review_length_words": [99, 99, 99],
        }
    )

    result = engineer_features(source)

    assert result["review_length_words"].tolist() == [2, 2, 0]
    assert result["review_length_chars"].tolist() == [12, 9, 0]
    assert result["skin_type_known"].tolist() == [True, False, False]
    assert result["rating_band"].tolist() == [
        "high (4-5)",
        "low (1-2)",
        "middle (3)",
    ]
    assert source["review_length_words"].tolist() == [99, 99, 99]


def test_rating_band_leaves_missing_and_out_of_range_values_unlabelled():
    result = engineer_features(
        pd.DataFrame({"rating": [1, 3, 5, 0, 6, np.nan]})
    )

    assert result["rating_band"].iloc[:3].tolist() == [
        "low (1-2)",
        "middle (3)",
        "high (4-5)",
    ]
    assert result["rating_band"].iloc[3:].isna().all()


def test_engineer_features_handles_missing_optional_columns():
    source = pd.DataFrame({"product_id": ["P1", "P2"]})

    result = engineer_features(source)

    assert result.columns.tolist() == ["product_id"]


def test_summary_uses_column_specific_sample_sizes_and_skips_absent_fields(
    tmp_path,
):
    source = pd.DataFrame(
        {
            "rating": [1, 5, np.nan],
            "review_text_clean": ["one", "", "three words"],
            "skin_type": ["dry", "unknown", "oily"],
        }
    )
    features = engineer_features(source)

    summary = build_summary(features, tmp_path / "sample.csv")

    assert summary["row_count"] == 3
    assert summary["analysis_sample_sizes"]["rating"] == 2
    assert summary["analysis_sample_sizes"]["review_length"] == 2
    assert summary["analysis_sample_sizes"]["skin_type_distribution"] == 3
    assert summary["analysis_sample_sizes"]["ratings_by_skin_type"] == 1
    assert summary["statistics"]["rating"]["counts"] == {"1": 1, "5": 1}
    assert summary["statistics"]["ratings_by_skin_type"] == {
        "dry": {"count": 1, "mean": 1.0, "median": 1.0}
    }
    assert "most_reviewed_products" not in summary["statistics"]


def test_load_reviews_normalizes_headers_and_rejects_empty_file(tmp_path):
    source = tmp_path / "reviews.csv"
    source.write_text(" Rating , Product_ID \n5,P1\n", encoding="utf-8")

    loaded = load_reviews(source)

    assert loaded.columns.tolist() == ["rating", "product_id"]
    source.write_text("rating,product_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no rows"):
        load_reviews(source)


def test_load_reviews_rejects_duplicate_normalized_headers(tmp_path):
    source = tmp_path / "reviews.csv"
    source.write_text("rating, Rating \n5,4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_reviews(source)


def test_chart_generation_skips_analyses_when_fields_are_absent(tmp_path):
    source = pd.DataFrame({"other_field": [1, 2]})
    summary = build_summary(source, tmp_path / "sample.csv")

    generated, skipped = create_charts(source, summary, tmp_path / "figures")

    assert [path.name for path in generated] == ["missing_values.png"]
    assert "rating_distribution" in skipped
    assert "review_length_distribution" in skipped
    assert "skin_type_distribution" in skipped
    assert "ratings_by_skin_type" in skipped
    assert "most_reviewed_products" in skipped
