"""
scripts/make_sample.py — generate synthetic Sephora-like sample CSV
====================================================================
Creates data/raw/reviews_sample.csv with ~200 rows that replicate the column
structure and known data quality issues of the real Sephora dataset.

Issues deliberately injected:
  - Null review_text (2 rows)
  - Blank review_text (2 rows)
  - Null rating (3 rows)
  - Non-numeric rating ("five", "great") (2 rows)
  - Out-of-range rating (0, 6) (2 rows)
  - Fractional ratings (3.7, 4.2) (2 rows)
  - Null skin_type (15 rows)
  - Unrecognised skin_type ("combo", "mixed", "sensitive/dry", "weirdvalue") (6 rows)
  - Exact duplicate reviews from same author (4 pairs = 8 rows, first kept)
  - Non-skincare primary_category rows ("Makeup", "Fragrance") (10 rows)
  - HTML tags in review_text (5 rows)
  - URLs in review_text (3 rows)
  - Shouted punctuation (!!!, ???) in review_text (4 rows)
  - Negation language that must survive (scattered through)
  - Missing optional columns (brand_name) for some rows
"""

import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "reviews_sample.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Base vocabulary
# -------------------------------------------------------------------

CLEAN_REVIEWS = [
    "This moisturiser is absolutely wonderful. My dry skin has never felt better.",
    "I do not recommend this serum. It broke me out terribly.",
    "Not sure about this one. It doesn't feel hydrating enough.",
    "Love this cleanser! It never leaves my skin feeling tight.",
    "The sunscreen doesn't leave a white cast, which is great for my skin tone.",
    "Won't repurchase. My skin reacted badly and the texture isn't pleasant.",
    "This toner is incredibly gentle — no irritation at all.",
    "I can't believe how quickly this eye cream reduced my dark circles.",
    "Doesn't work for oily skin types. Would not buy again.",
    "Amazing retinol treatment. My skin texture has improved noticeably.",
    "The face wash is too harsh for sensitive skin. Avoid if easily irritated.",
    "Perfect for combination skin. Not too heavy, not too light.",
    "No fragrance, no parabens, and no nasty chemicals. Exactly what I needed.",
    "My skin isn't tolerating this well. Redness after each use.",
    "Never been so impressed with a drugstore moisturiser at this price.",
    "This SPF is great but the sunscreen formula doesn't blend well.",
    "I wasn't expecting much but it really does work for dry patches.",
    "Can't stop using this exfoliant. My dull skin has completely transformed.",
    "Honestly not worth the price. Didn't see any difference.",
    "The formula is rich but doesn't feel greasy — perfect for night time.",
    "Wouldn't buy this again. The smell is overwhelming.",
    "No breakouts since switching to this cleanser. Highly recommend.",
    "My combination skin loves this. Not drying, not oily.",
    "This doesn't work as advertised. Very disappointed with the results.",
    "Can't believe I didn't discover this serum sooner. Incredible results!",
]

SKINCARE_CATEGORIES = ["Skincare"] * 18 + ["Makeup", "Fragrance"]
PRODUCTS = [f"P{str(i).zfill(4)}" for i in range(1, 31)]
BRANDS = ["CeraVe", "La Roche-Posay", "Neutrogena", "The Ordinary",
          "Paula's Choice", "Cetaphil", "Drunk Elephant", "Tatcha"]
SKIN_TYPES_VALID = ["dry", "oily", "combination", "normal", "sensitive"]
AUTHORS = [f"U{str(i).zfill(4)}" for i in range(1, 80)]

# -------------------------------------------------------------------
# Build the base rows (180 clean rows)
# -------------------------------------------------------------------

rows = []
for i in range(180):
    text = CLEAN_REVIEWS[i % len(CLEAN_REVIEWS)]
    rows.append(
        {
            "author_id": AUTHORS[i % len(AUTHORS)],
            "product_id": PRODUCTS[i % len(PRODUCTS)],
            "product_name": f"Product {PRODUCTS[i % len(PRODUCTS)]}",
            "brand_name": BRANDS[i % len(BRANDS)],
            "primary_category": SKINCARE_CATEGORIES[i % len(SKINCARE_CATEGORIES)],
            "secondary_category": random.choice(
                ["Moisturisers", "Serums", "Cleansers", "SPF", "Treatments"]
            ),
            "review_title": f"Review {i}",
            "review_text": text,
            "rating": random.choice([1, 2, 3, 4, 5]),
            "skin_type": SKIN_TYPES_VALID[i % len(SKIN_TYPES_VALID)],
            "skin_tone": random.choice(["fair", "medium", "deep", None]),
            "eye_color": random.choice(["blue", "brown", "green", None]),
            "hair_color": random.choice(["blonde", "brunette", "black", None]),
            "helpfulness": round(random.uniform(0, 1), 2),
            "price_usd": round(random.uniform(5, 120), 2),
            "loves_count": random.randint(0, 5000),
            "submission_time": f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "is_recommended": random.choice([0, 1]),
        }
    )

# -------------------------------------------------------------------
# Inject dirty rows
# -------------------------------------------------------------------

def _base_row(i: int = 0) -> dict:
    return rows[i].copy()


# 2 null review_text
for _ in range(2):
    r = _base_row()
    r["author_id"] = f"UNULL{_}"
    r["review_text"] = None
    rows.append(r)

# 2 blank review_text
for i in range(2):
    r = _base_row()
    r["author_id"] = f"UBLNK{i}"
    r["review_text"] = "   "
    rows.append(r)

# 3 null rating
for i in range(3):
    r = _base_row()
    r["author_id"] = f"URNUL{i}"
    r["rating"] = None
    rows.append(r)

# 2 non-numeric rating
for val in ["five", "great"]:
    r = _base_row()
    r["author_id"] = f"URSTR{val}"
    r["rating"] = val
    rows.append(r)

# 2 out-of-range rating
for val in [0, 6]:
    r = _base_row()
    r["author_id"] = f"UROOR{val}"
    r["rating"] = val
    rows.append(r)

# 2 fractional ratings
for val in [3.7, 4.2]:
    r = _base_row()
    r["author_id"] = f"URFRAC{int(val*10)}"
    r["rating"] = val
    rows.append(r)

# 15 null skin_type
for i in range(15):
    r = _base_row(i)
    r["author_id"] = f"USKN{i:02d}"
    r["skin_type"] = None
    rows.append(r)

# 6 unrecognised skin_type
for val in ["combo", "mixed", "sensitive/dry", "weirdvalue", "Combination/Oily", "all"]:
    r = _base_row()
    r["author_id"] = f"USKNU{val[:4]}"
    r["skin_type"] = val
    rows.append(r)

# 4 duplicate pairs (exact same author_id + review_text)
for i in range(4):
    original = rows[i].copy()
    dup = original.copy()
    # Duplicate has same author and text but different row position
    rows.append(dup)

# 5 HTML in review_text
html_reviews = [
    "This is <b>amazing</b>! Best product ever.",
    "<i>Love</i> the texture — <br>super lightweight.",
    "Visit <a href='#'>our site</a> for more details.",
    "The <strong>packaging</strong> is so pretty.",
    "Didn't work for me. <ul><li>No hydration</li><li>Greasy</li></ul>",
]
for i, text in enumerate(html_reviews):
    r = _base_row()
    r["author_id"] = f"UHTML{i}"
    r["review_text"] = text
    r["primary_category"] = "Skincare"
    rows.append(r)

# 3 URLs in review_text
url_reviews = [
    "Check https://skincareaddicts.com for my full review!",
    "More info at www.beautyblog.com/review. Loved this product.",
    "Buy it from https://sephora.com — it's so worth it.",
]
for i, text in enumerate(url_reviews):
    r = _base_row()
    r["author_id"] = f"UURL{i}"
    r["review_text"] = text
    r["primary_category"] = "Skincare"
    rows.append(r)

# 4 shouted punctuation
shout_reviews = [
    "AMAZING!!! I can't live without this!!!",
    "TERRIBLE??? Didn't work at all???",
    "Not. Even. Close. to what I expected...",
    "SO GOOD!!! Will never go back to my old routine!!!",
]
for i, text in enumerate(shout_reviews):
    r = _base_row()
    r["author_id"] = f"USHOUT{i}"
    r["review_text"] = text
    r["primary_category"] = "Skincare"
    rows.append(r)

# -------------------------------------------------------------------
# Assemble DataFrame and shuffle
# -------------------------------------------------------------------

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

df.to_csv(OUT_PATH, index=False)

print(f"Sample CSV written to {OUT_PATH}")
print(f"Total rows: {len(df)}")
print(f"Columns   : {list(df.columns)}")
print(f"\nIntrospect issues injected:")
print(f"  null review_text   : {df['review_text'].isna().sum()}")
print(f"  blank review_text  : {(df['review_text'].fillna('X').str.strip() == '').sum()}")
print(f"  null rating        : {df['rating'].isna().sum()}")
print(f"  null skin_type     : {df['skin_type'].isna().sum()}")
