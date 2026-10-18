"""Week 4 exploratory data analysis for cleaned customer reviews.

Run from the repository root:
    python src/eda.py

The default input is the real Week 3 sample output. Pass --input to analyze a
different cleaned file. Existing outputs are protected unless --overwrite is
provided.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "processed" / "reviews_clean_sample.csv"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "reviews_eda_features.csv"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "eda_summary.json"
DEFAULT_FIGURES = ROOT / "figures"
DEFAULT_REPORT = ROOT / "reports" / "Week_4_Report.docx"

CHART_NAMES = {
    "rating_distribution": "rating_distribution.png",
    "missing_values": "missing_values.png",
    "review_length_distribution": "review_length_distribution.png",
    "skin_type_distribution": "skin_type_distribution.png",
    "ratings_by_skin_type": "ratings_by_skin_type.png",
    "most_reviewed_products": "most_reviewed_products.png",
}

FEATURE_DEFINITIONS = {
    "review_length_words": (
        "Word count in review_text_clean, computed by whitespace-separated tokens."
    ),
    "review_length_chars": "Character count in review_text_clean.",
    "skin_type_known": (
        "True when skin_type is present and is not the canonical value 'unknown'; "
        "False otherwise."
    ),
    "rating_band": (
        "Descriptive label derived from rating: low (1-2), middle (3), or high "
        "(4-5). This is an outcome summary, not a predictor."
    ),
}

log = logging.getLogger(__name__)


def load_reviews(input_path: Path) -> pd.DataFrame:
    """Load a cleaned CSV and normalize header whitespace/case."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Cleaned review CSV not found: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    normalized = {column: str(column).strip().lower() for column in df.columns}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Column normalization would create duplicate column names.")
    df = df.rename(columns=normalized)
    if df.empty:
        raise ValueError(f"Cleaned review CSV contains no rows: {input_path}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive, non-predictive features without changing the input."""
    features = df.copy()

    if "review_text_clean" in features.columns:
        text = features["review_text_clean"].fillna("").astype(str)
        features["review_length_words"] = (
            text.str.split().str.len().fillna(0).astype("int64")
        )
        features["review_length_chars"] = text.str.len().astype("int64")

    if "skin_type" in features.columns:
        skin_type = features["skin_type"].fillna("").astype(str).str.strip().str.lower()
        features["skin_type_known"] = skin_type.ne("") & skin_type.ne("unknown")

    if "rating" in features.columns:
        rating = pd.to_numeric(features["rating"], errors="coerce")
        valid = rating.between(1, 5, inclusive="both")
        bands = np.select(
            [
                valid & rating.isin([1, 2]),
                valid & rating.eq(3),
                valid & rating.isin([4, 5]),
            ],
            ["low (1-2)", "middle (3)", "high (4-5)"],
            default="",
        )
        features["rating_band"] = pd.Series(bands, index=features.index).replace(
            "", pd.NA
        ).astype("string")

    return features


def _number(value: Any) -> Optional[float | int]:
    """Convert a numeric result to JSON-safe Python values."""
    if pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _distribution(series: pd.Series) -> dict[str, int]:
    counts = series.astype("string").fillna("unknown").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def build_summary(
    df: pd.DataFrame,
    input_path: Path,
    input_columns: Optional[dict[str, str]] = None,
    input_missing: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Calculate actual dataset, missingness, and analysis-specific summaries."""
    row_count = int(len(df))
    source_columns = input_columns or {
        str(column): str(dtype) for column, dtype in df.dtypes.items()
    }
    missing_count = (
        pd.Series(input_missing, dtype="int64")
        if input_missing is not None
        else df.isna().sum()
    )
    missing_summary = {
        column: {
            "count": int(count),
            "percent": _number(count / row_count * 100) if row_count else None,
        }
        for column, count in missing_count.items()
    }

    summary: dict[str, Any] = {
        "input_file": str(input_path),
        "row_count": row_count,
        "column_count": int(len(source_columns)),
        "columns": source_columns,
        "feature_columns": {
            str(column): str(dtype) for column, dtype in df.dtypes.items()
        },
        "missing_values": missing_summary,
        "analysis_sample_sizes": {},
        "statistics": {},
        "feature_definitions": FEATURE_DEFINITIONS,
        "interpretation_note": (
            "Descriptive statistics represent only the specified input file. "
            "They do not validate skin concerns, product suitability, causality, "
            "or population-level differences."
        ),
    }

    if "rating" in df.columns:
        rating = pd.to_numeric(df["rating"], errors="coerce")
        valid_rating = rating[np.isfinite(rating) & rating.between(1, 5)]
        summary["analysis_sample_sizes"]["rating"] = int(valid_rating.count())
        summary["statistics"]["rating"] = {
            "count": int(valid_rating.count()),
            "mean": _number(valid_rating.mean()),
            "median": _number(valid_rating.median()),
            "std_sample": _number(valid_rating.std()),
            "minimum": _number(valid_rating.min()),
            "maximum": _number(valid_rating.max()),
            "counts": {
                str(_number(key)): int(value)
                for key, value in valid_rating.value_counts().sort_index().items()
            },
        }
        band_counts = (
            df["rating_band"].value_counts(dropna=False).to_dict()
            if "rating_band" in df.columns
            else {}
        )
        summary["statistics"]["rating_band_counts"] = {
            str(key): int(value)
            for key, value in band_counts.items()
            if not pd.isna(key)
        }

    summary["analysis_sample_sizes"]["missing_values"] = row_count
    summary["statistics"]["missing_value_columns_with_missing_data"] = int(
        missing_count.gt(0).sum()
    )

    if "review_text_clean" in df.columns:
        clean_text = df["review_text_clean"].fillna("").astype(str)
        nonempty = clean_text.str.strip().ne("")
        words = clean_text.str.split().str.len().astype("int64")
        chars = clean_text.str.len().astype("int64")
        summary["analysis_sample_sizes"]["review_length"] = int(nonempty.sum())
        summary["statistics"]["review_length_words"] = {
            key: _number(value)
            for key, value in words[nonempty].describe().items()
        }
        summary["statistics"]["review_length_chars"] = {
            key: _number(value)
            for key, value in chars[nonempty].describe().items()
        }

    if "skin_type" in df.columns:
        skin_type = df["skin_type"].fillna("unknown").astype(str).str.strip()
        known = skin_type.str.lower().ne("") & skin_type.str.lower().ne("unknown")
        summary["analysis_sample_sizes"]["skin_type_distribution"] = row_count
        summary["statistics"]["skin_type_counts"] = _distribution(skin_type)
        valid_rating = pd.Series(False, index=df.index)
        if "rating" in df.columns:
            rating = pd.to_numeric(df["rating"], errors="coerce")
            valid_rating = rating.between(1, 5, inclusive="both")
        summary["analysis_sample_sizes"]["ratings_by_skin_type"] = int(
            (known & valid_rating).sum()
        )
        if "rating" in df.columns:
            rating = pd.to_numeric(df["rating"], errors="coerce")
            rating = rating.where(rating.between(1, 5, inclusive="both"))
            grouped = pd.DataFrame(
                {"skin_type": skin_type, "rating": rating}
            ).loc[known]
            grouped = grouped.dropna(subset=["rating"])
            group_stats = grouped.groupby("skin_type")["rating"].agg(
                ["count", "mean", "median"]
            )
            summary["statistics"]["ratings_by_skin_type"] = {
                str(index): {
                    "count": int(row["count"]),
                    "mean": _number(row["mean"]),
                    "median": _number(row["median"]),
                }
                for index, row in group_stats.iterrows()
            }

    if "product_id" in df.columns:
        product_ids = df["product_id"].dropna().astype(str)
        summary["analysis_sample_sizes"]["product_counts"] = int(
            product_ids.shape[0]
        )
        summary["statistics"]["unique_products"] = int(product_ids.nunique())
        product_counts = product_ids.value_counts().head(10)
        top_products = []
        for product_id, count in product_counts.items():
            product_name: Optional[str] = None
            if "product_name" in df.columns:
                names = df.loc[
                    df["product_id"].astype(str).eq(product_id), "product_name"
                ].dropna()
                if not names.empty:
                    product_name = str(names.iloc[0])
            top_products.append(
                {
                    "product_id": str(product_id),
                    "product_name": product_name,
                    "review_count": int(count),
                }
            )
        summary["statistics"]["most_reviewed_products"] = top_products

    return summary


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved: %s", path)


def create_charts(
    df: pd.DataFrame,
    summary: dict[str, Any],
    figures_dir: Path,
) -> tuple[list[Path], dict[str, str]]:
    """Create only plots supported by columns and observations in the input."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    generated: list[Path] = []
    skipped: dict[str, str] = {}

    def save(name: str, fig: plt.Figure) -> None:
        path = figures_dir / CHART_NAMES[name]
        _save_figure(fig, path)
        generated.append(path)

    if "rating" in df.columns:
        ratings = pd.to_numeric(df["rating"], errors="coerce")
        ratings = ratings[ratings.between(1, 5)]
        if not ratings.empty:
            counts = ratings.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(counts.index.to_numpy(dtype=int), counts.values, color="#4C78A8")
            ax.set(
                title=f"Rating distribution (n={len(ratings):,})",
                xlabel="Rating (1–5)",
                ylabel="Reviews",
                xticks=[1, 2, 3, 4, 5],
            )
            save("rating_distribution", fig)
        else:
            skipped["rating_distribution"] = "No numeric ratings in the 1–5 range."
    else:
        skipped["rating_distribution"] = "The rating column is absent."

    missing = summary["missing_values"]
    missing_counts = pd.Series(
        {column: values["count"] for column, values in missing.items()},
        dtype="int64",
    )
    fig, ax = plt.subplots(figsize=(max(8, len(missing_counts) * 0.48), 5))
    colors = ["#E45756" if value else "#72B7B2" for value in missing_counts.values]
    sns.barplot(x=missing_counts.index, y=missing_counts.values, palette=colors, hue=missing_counts.index, legend=False, ax=ax)
    ax.tick_params(axis="x", labelrotation=65)
    ax.set(title=f"Missing values by column (n={len(df):,})", xlabel="Column", ylabel="Missing rows")
    save("missing_values", fig)

    if "review_text_clean" in df.columns:
        text = df["review_text_clean"].fillna("").astype(str)
        lengths = text.str.split().str.len()
        lengths = lengths[text.str.strip().ne("")]
        if not lengths.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.histplot(lengths, bins=30, color="#59A14F", ax=ax)
            ax.set(title=f"Review length distribution (n={len(lengths):,})", xlabel="Words per cleaned review", ylabel="Reviews")
            save("review_length_distribution", fig)
        else:
            skipped["review_length_distribution"] = "No non-empty cleaned review text."
    else:
        skipped["review_length_distribution"] = "The review_text_clean column is absent."

    if "skin_type" in df.columns:
        skin_types = df["skin_type"].fillna("unknown").astype(str).str.strip()
        counts = skin_types.replace("", "unknown").value_counts()
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.barplot(x=counts.index, y=counts.values, color="#F28E2B", ax=ax)
            ax.set(title=f"Reported skin-type categories (n={len(df):,})", xlabel="Reported skin type", ylabel="Reviews")
            ax.tick_params(axis="x", labelrotation=20)
            save("skin_type_distribution", fig)
        else:
            skipped["skin_type_distribution"] = "No skin-type values are available."
    else:
        skipped["skin_type_distribution"] = "The skin_type column is absent."

    if {"rating", "skin_type"}.issubset(df.columns):
        rating = pd.to_numeric(df["rating"], errors="coerce")
        skin = df["skin_type"].fillna("").astype(str).str.strip()
        known = skin.str.lower().ne("") & skin.str.lower().ne("unknown")
        plot_data = pd.DataFrame({"rating": rating, "skin_type": skin}).loc[known]
        plot_data = plot_data.loc[
            plot_data["rating"].between(1, 5, inclusive="both")
        ]
        plot_data = plot_data.dropna(subset=["rating"])
        if not plot_data.empty:
            order = plot_data["skin_type"].value_counts().index.tolist()
            counts = plot_data["skin_type"].value_counts()
            fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.6), 5))
            sns.boxplot(data=plot_data, x="skin_type", y="rating", order=order, color="#A0CBE8", showfliers=False, ax=ax)
            ax.set_xticks(
                np.arange(len(order)),
                labels=[f"{kind}\n(n={counts[kind]:,})" for kind in order],
            )
            ax.set(title=f"Rating distributions by reported skin type (n={len(plot_data):,}; unknown excluded)", xlabel="Self-reported skin type", ylabel="Rating (1–5)", ylim=(0.5, 5.5))
            save("ratings_by_skin_type", fig)
        else:
            skipped["ratings_by_skin_type"] = "No known skin types with ratings are available."
    else:
        missing_columns = sorted({"rating", "skin_type"} - set(df.columns))
        skipped["ratings_by_skin_type"] = (
            "Required column(s) absent: " + ", ".join(missing_columns)
        )

    if "product_id" in df.columns:
        products = df["product_id"].dropna().astype(str).value_counts().head(10)
        if not products.empty:
            labels = products.index.tolist()
            if "product_name" in df.columns:
                label_map = (
                    df.dropna(subset=["product_id"])
                    .assign(product_id=lambda frame: frame["product_id"].astype(str))
                    .drop_duplicates("product_id")
                    .set_index("product_id")["product_name"]
                    .to_dict()
                )
                labels = [
                    f"{product_id} — {str(label_map.get(product_id, '')).strip()[:36]}"
                    .rstrip(" —")
                    for product_id in products.index
                ]
            fig, ax = plt.subplots(figsize=(10, max(4.5, len(products) * 0.55)))
            sns.barplot(x=products.values, y=labels, color="#B279A2", ax=ax)
            ax.set(title=f"Most-reviewed products (top {len(products)}; n={int(products.sum()):,})", xlabel="Reviews in input", ylabel="Product")
            save("most_reviewed_products", fig)
        else:
            skipped["most_reviewed_products"] = "No non-missing product identifiers."
    else:
        skipped["most_reviewed_products"] = "The product_id column is absent."

    return generated, skipped


def write_report(
    summary: dict[str, Any],
    chart_paths: list[Path],
    report_path: Path,
) -> None:
    """Create a DOCX report with sample-specific results and embedded charts."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    styles = document.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10)
    for style_name in ("Heading 1", "Heading 2"):
        styles[style_name].font.name = "Arial"
        styles[style_name].font.color.rgb = RGBColor(31, 78, 121)

    title = document.add_heading(
        "Week 4: Exploratory Data Analysis and Feature Engineering", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph(
        "Analyzing Customer Reviews for Personalized Skincare Recommendations"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].bold = True
    date_line = document.add_paragraph("Beauty & Wellness Data Science Internship")
    date_line.alignment = WD_ALIGN_PARAGRAPH.CENTER

    row_count = summary["row_count"]
    rating = summary["statistics"].get("rating")
    length = summary["statistics"].get("review_length_words")
    skin_counts = summary["statistics"].get("skin_type_counts", {})
    product_counts = summary["statistics"].get("most_reviewed_products", [])
    rating_groups = summary["statistics"].get("ratings_by_skin_type", {})

    document.add_heading("Objective and scope", level=1)
    document.add_paragraph(
        "Describe the cleaned review sample using available ratings, review text, "
        "self-reported skin-type categories, and product identifiers. The analysis "
        "is exploratory: it does not test causal effects, validate skin concerns, "
        "or establish that a product is suitable for a particular person."
    )
    document.add_paragraph(
        f"Input: {summary['input_file']}. The file contains {row_count:,} rows and "
        f"{summary['column_count']} columns. These findings apply only to this "
        "input file, not to the full Sephora review corpus."
    )

    document.add_heading("Methodology", level=1)
    for item in [
        "Loaded the actual Week 3 cleaned sample without modifying it.",
        "Calculated per-column missing-value counts and percentages against all input rows.",
        "Summarized ratings using valid integer values from 1 through 5; reported count, mean, median, sample standard deviation, minimum, maximum, and frequency counts.",
        "Recomputed word and character lengths from review_text_clean for consistent descriptive analysis.",
        "Included unknown/missing profiles in the skin-type count chart; excluded unknown profiles from ratings-by-skin-type summaries.",
        "Counted product_id values and attached product names for display when available.",
        "Generated only charts supported by the available input columns. Each analysis records its own usable sample size in the JSON output.",
    ]:
        document.add_paragraph(item, style="List Bullet")

    document.add_heading("Preprocessing and analysis flow", level=1)
    stages = [
        "Week 3 cleaned sample CSV",
        "Schema and missing-value inventory",
        "Rating, review-length, skin-type and product summaries",
        "Descriptive features (length, known-profile flag, rating band)",
        "Charts + machine-readable JSON summary",
    ]
    flow = document.add_table(rows=0, cols=1)
    flow.style = "Table Grid"
    for index, stage in enumerate(stages):
        cell = flow.add_row().cells[0]
        cell.text = stage
        if index < len(stages) - 1:
            arrow = document.add_paragraph("↓")
            arrow.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("Actual findings", level=1)
    if rating:
        document.add_paragraph(
            f"Ratings were available for n={rating['count']:,} reviews. The mean "
            f"was {rating['mean']:.3f}, the median was {rating['median']}, and "
            f"observed values ranged from {rating['minimum']} to {rating['maximum']}. "
            f"Counts by rating: {rating['counts']}."
        )
    if length:
        document.add_paragraph(
            f"Cleaned review length (n={length['count']:,}) averaged "
            f"{length['mean']:.2f} words (median {length['50%']:.0f}; "
            f"range {length['min']:.0f}–{length['max']:.0f})."
        )

    missing = [
        (column, values["count"], values["percent"])
        for column, values in summary["missing_values"].items()
        if values["count"] > 0
    ]
    document.add_paragraph(
        f"Missing values occurred in {len(missing)} of "
        f"{summary['column_count']} columns. The input contained {row_count:,} rows."
    )
    if missing:
        table = document.add_table(rows=1, cols=3)
        table.style = "Light Shading Accent 1"
        for cell, value in zip(table.rows[0].cells, ["Column", "Missing rows", "Missing (%)"]):
            cell.text = value
        for column, count, percent in missing:
            cells = table.add_row().cells
            cells[0].text = str(column)
            cells[1].text = f"{count:,}"
            cells[2].text = f"{percent:.2f}"

    if skin_counts:
        document.add_paragraph(
            f"Reported skin-type category counts for n={summary['analysis_sample_sizes']['skin_type_distribution']:,}: "
            + ", ".join(f"{name}: {count:,}" for name, count in skin_counts.items())
            + ". The unknown group is retained in this distribution."
        )
    if rating_groups:
        document.add_paragraph(
            "Among reviews with a known self-reported skin type and usable rating "
            f"(n={summary['analysis_sample_sizes']['ratings_by_skin_type']:,}), "
            "descriptive mean ratings were: "
            + "; ".join(
                f"{name} n={values['count']:,}, mean={values['mean']:.3f}"
                for name, values in rating_groups.items()
            )
            + ". These are within-sample descriptions, not evidence of product "
            "suitability or statistically validated differences."
        )
    if product_counts:
        document.add_paragraph(
            f"Product identifiers were available for n={summary['analysis_sample_sizes']['product_counts']:,} "
            f"reviews, covering {summary['statistics']['unique_products']} unique products."
        )
        product_table = document.add_table(rows=1, cols=3)
        product_table.style = "Light Shading Accent 1"
        for cell, value in zip(
            product_table.rows[0].cells, ["Product ID", "Product name", "Reviews in sample"]
        ):
            cell.text = value
        for product in product_counts:
            cells = product_table.add_row().cells
            cells[0].text = product["product_id"]
            cells[1].text = product.get("product_name") or "Not available"
            cells[2].text = f"{product['review_count']:,}"

    document.add_heading("Input schema and data quality", level=1)
    schema_table = document.add_table(rows=1, cols=2)
    schema_table.style = "Light Shading Accent 1"
    schema_table.rows[0].cells[0].text = "Input column"
    schema_table.rows[0].cells[1].text = "Pandas dtype"
    for column, dtype in summary["columns"].items():
        cells = schema_table.add_row().cells
        cells[0].text = str(column)
        cells[1].text = str(dtype)

    document.add_heading("Charts and captions", level=1)
    captions = {
        "rating_distribution.png": (
            "Rating distribution. Bars show observed counts for integer ratings "
            "1–5; n is the number of valid ratings."
        ),
        "missing_values.png": (
            "Missing-value count by input column. Counts are shown rather than "
            "imputed values; optional missing data were retained."
        ),
        "review_length_distribution.png": (
            "Word count per non-empty cleaned review. Word count is descriptive "
            "and does not indicate review quality."
        ),
        "skin_type_distribution.png": (
            "Self-reported skin-type categories. Unknown values remain visible; "
            "the categories are not independently verified."
        ),
        "ratings_by_skin_type.png": (
            "Rating distributions by known self-reported skin type; unknown "
            "profiles are excluded. This chart is descriptive and confounded by "
            "the sample's product concentration."
        ),
        "most_reviewed_products.png": (
            "Products with the highest review counts within this input sample; "
            "counts do not represent product-wide totals."
        ),
    }
    for chart in chart_paths:
        document.add_picture(str(chart), width=Inches(6.1))
        caption = document.add_paragraph(captions.get(chart.name, chart.name))
        caption.style = "Caption"

    document.add_heading("Feature definitions and leakage safeguards", level=1)
    feature_text = {
        "review_length_words": (
            "Whitespace-separated token count from review_text_clean. Recomputed "
            "for the EDA output; descriptive text metadata."
        ),
        "review_length_chars": (
            "Character count of review_text_clean. Descriptive text metadata."
        ),
        "skin_type_known": (
            "Boolean flag: True when skin_type is non-empty and not 'unknown'; "
            "False otherwise. It indicates reported profile availability, not a "
            "validated skin characteristic."
        ),
        "rating_band": (
            "low (1–2), middle (3), or high (4–5), derived directly from rating. "
            "This is an outcome label for summaries only and must not be used as "
            "a predictor when rating or recommendation is the target."
        ),
    }
    for feature, definition in feature_text.items():
        document.add_paragraph(f"{feature}: {definition}", style="List Bullet")
    document.add_paragraph(
        "The Week 3 cleaned CSV remains unchanged. Engineered columns are written "
        "to a separate EDA feature CSV. No skin-concern keyword, sentiment, "
        "medical, or suitability feature was created."
    )

    document.add_heading("Limitations", level=1)
    for item in [
        "The input is the first 1,000 rows in file order from one review chunk, reduced to 999 by Week 3 duplicate removal. It is not a random or representative sample of all Sephora reviews.",
        "The sample is extremely concentrated: 998 of 999 rows refer to one product and one row to another. Skin-type rating summaries are therefore dominated by that product and cannot establish general product suitability or independent skin-type effects.",
        "Skin type is self-reported; 102 rows are categorized as unknown. The unknown value does not identify a true underlying profile.",
        "Ratings and review text are self-selected and may be biased. Descriptive rating differences are not causal or inferential.",
        "The full raw review corpus has not yet been cleaned and analyzed. Full-dataset results may differ.",
        "No evidence in this analysis validates customer skin concerns, product compatibility, or medical outcomes. The work is not dermatological advice.",
    ]:
        document.add_paragraph(item, style="List Bullet")

    document.add_heading("Conclusion", level=1)
    document.add_paragraph(
        f"This EDA pipeline produced sample-specific summaries and {len(chart_paths)} "
        "charts from the available cleaned fields. It identifies a rating "
        "distribution, missing optional data, review lengths, reported skin-type "
        "categories, and product concentration. Given the non-representative "
        "sample and single-product concentration, these results do not validate "
        "personalized recommendations. Run EDA on the full cleaned corpus before "
        "drawing broader conclusions."
    )

    document.add_heading("References", level=1)
    for reference in [
        "Sephora Products and Skincare Reviews dataset, Kaggle: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews (local sample inspected; license not verified).",
        "pandas documentation: https://pandas.pydata.org/docs/",
        "NumPy documentation: https://numpy.org/doc/stable/",
        "Matplotlib documentation: https://matplotlib.org/stable/",
        "seaborn documentation: https://seaborn.pydata.org/",
    ]:
        document.add_paragraph(reference, style="List Bullet")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(report_path)
    log.info("Week 4 report saved: %s", report_path)


def run_eda(
    input_path: Path = DEFAULT_INPUT,
    figures_dir: Path = DEFAULT_FIGURES,
    summary_path: Path = DEFAULT_SUMMARY,
    features_path: Path = DEFAULT_FEATURES,
    report_path: Path = DEFAULT_REPORT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run EDA, save descriptive feature data, charts, and a JSON summary."""
    df = load_reviews(input_path)
    input_columns = {str(column): str(dtype) for column, dtype in df.dtypes.items()}
    input_missing = {
        str(column): int(count) for column, count in df.isna().sum().items()
    }
    features = engineer_features(df)
    summary = build_summary(
        features,
        input_path,
        input_columns=input_columns,
        input_missing=input_missing,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    expected_chart_paths = [
        figures_dir / filename for filename in CHART_NAMES.values()
    ]
    targets = [summary_path, features_path, report_path, *expected_chart_paths]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing EDA output(s); pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    generated, skipped = create_charts(features, summary, figures_dir)
    summary["generated_charts"] = [str(path) for path in generated]
    summary["skipped_analyses"] = skipped

    features_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False, allow_nan=False)
    write_report(summary, generated, report_path)

    print(f"Input: {input_path}")
    print(f"Rows: {len(df):,}; columns: {len(df.columns)}")
    print(f"Charts generated: {len(generated)}")
    for chart in generated:
        print(f"  {chart}")
    for analysis, reason in skipped.items():
        print(f"Skipped {analysis}: {reason}")
    print(f"Feature data: {features_path}")
    print(f"EDA summary: {summary_path}")
    print(f"Week 4 report: {report_path}")
    print(json.dumps(summary["statistics"], indent=2, ensure_ascii=False))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 4 EDA for a cleaned customer review CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Cleaned CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES,
        help="Directory for chart PNG files.",
    )
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY,
        help="Path for the machine-readable JSON summary.",
    )
    parser.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES,
        help="Path for the derived feature CSV.",
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT,
        help="Path for the Week 4 DOCX report.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing EDA outputs at the selected output paths.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _parse_args()
    run_eda(
        input_path=arguments.input,
        figures_dir=arguments.figures_dir,
        summary_path=arguments.summary,
        features_path=arguments.features,
        report_path=arguments.report,
        overwrite=arguments.overwrite,
    )
