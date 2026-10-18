"""Tests for Week 5 target preparation and leakage-safe model pipelines."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import (
    TARGET,
    build_models,
    calculate_metrics,
    collect_labeled_sample,
    prepare_xy,
)


def test_prepare_xy_excludes_target_and_rating_predictors():
    sample = pd.DataFrame(
        {
            "is_recommended": [1, 0, np.nan, 1],
            "review_text_clean": ["great product", "not good", "excluded", ""],
            "skin_type": ["dry", "unknown", "oily", "normal"],
            "author_id": ["A", "B", "C", "D"],
            "rating": [5, 1, 3, 4],
        }
    )

    x, y, groups = prepare_xy(sample)

    assert x.columns.tolist() == ["review_text_clean", "skin_type"]
    assert TARGET not in x.columns
    assert "rating" not in x.columns
    assert y.tolist() == [1, 0]
    assert groups.tolist() == ["A", "B"]


def test_collect_labeled_sample_is_reproducible_and_class_proportional(tmp_path):
    source = tmp_path / "reviews.csv"
    rows = [
        {
            "is_recommended": label,
            "review_text_clean": f"review text number {index}",
            "skin_type": "dry" if index % 2 else "unknown",
            "author_id": f"author-{index}",
            "rating": 4 if label else 2,
        }
        for index, label in enumerate([0] * 20 + [1] * 80)
    ]
    pd.DataFrame(rows).to_csv(source, index=False)

    first, metadata = collect_labeled_sample(
        source, sample_size=10, chunk_size=13, random_state=7
    )
    second, _ = collect_labeled_sample(
        source, sample_size=10, chunk_size=13, random_state=7
    )

    assert first.equals(second)
    assert first["is_recommended"].value_counts().sort_index().to_dict() == {
        0: 2,
        1: 8,
    }
    assert metadata["input_rows"] == 100
    assert metadata["valid_labeled_rows"] == 100


def test_collect_sample_requires_model_columns(tmp_path):
    source = tmp_path / "reviews.csv"
    pd.DataFrame({"rating": [5]}).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Required modeling columns are missing"):
        collect_labeled_sample(source, sample_size=10)


def test_logistic_pipeline_fits_vocabulary_on_training_data_only():
    x_train = pd.DataFrame(
        {
            "review_text_clean": [
                "hydrating gentle formula",
                "hydrating soft formula",
                "irritating drying formula",
                "irritating harsh formula",
            ],
            "skin_type": ["dry", "dry", "oily", "unknown"],
        }
    )
    y_train = pd.Series([1, 1, 0, 0])
    x_test = pd.DataFrame(
        {
            "review_text_clean": ["unseenkeyword hydrating formula"],
            "skin_type": ["new-profile"],
        }
    )

    model = build_models(max_text_features=50)["logistic_regression"]
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    vocabulary = model.named_steps["features"].named_transformers_[
        "text"
    ].vocabulary_

    assert prediction.tolist() in ([0], [1])
    assert "unseenkeyword" not in vocabulary


def test_calculate_metrics_reports_positive_class_and_auc():
    result = calculate_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        np.array([0.1, 0.6, 0.7, 0.9]),
    )

    assert result["accuracy"] == 0.75
    assert result["precision_positive"] == pytest.approx(2 / 3)
    assert result["recall_positive"] == 1.0
    assert result["roc_auc"] == 1.0
