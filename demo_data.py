"""A small synthetic dataset so the app can be tried without uploading anything."""

import numpy as np
import pandas as pd

_POSITIVE = [
    "Absolutely loved it, great quality and fast delivery.",
    "Excellent value, works perfectly and looks amazing.",
    "Really happy with this purchase, would buy again.",
    "Fantastic product, the best I have owned so far.",
    "Great design and very easy to use, highly recommended.",
    "Impressed by how well it works, worth every penny.",
]
_NEGATIVE = [
    "Terrible quality, it broke after two days. Total waste of money.",
    "Awful experience, the product was not as described.",
    "Very disappointed, it stopped working and support never replied.",
    "Worst purchase I have made, poor build and bad instructions.",
    "Not good at all, cheap materials and it arrived late.",
    "Do not buy this, it is overpriced and frustrating to use.",
]
_FILLER = ["", " Shipping took a while.", " The box was a bit damaged.", " I ordered it for my brother.", " Colour was as pictured."]


def make_demo_dataframe(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Product reviews with text, a sentiment label, numbers, dates and categories."""
    rng = np.random.default_rng(seed)
    positive = rng.random(n) < 0.62
    product = rng.choice(["Headphones", "Backpack", "Desk lamp", "Water bottle", "Keyboard"], n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    region = rng.choice(["North", "South", "East", "West"], n)
    base_price = {"Headphones": 80, "Backpack": 55, "Desk lamp": 35, "Water bottle": 20, "Keyboard": 60}
    price = np.array([base_price[p] for p in product]) * rng.normal(1, 0.15, n)
    price[rng.choice(n, 6, replace=False)] *= 4  # a few outliers
    rating = np.clip(np.where(positive, rng.normal(4.4, 0.7, n), rng.normal(1.9, 0.8, n)).round(), 1, 5)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D")
    text = [
        rng.choice(_POSITIVE if pos else _NEGATIVE) + rng.choice(_FILLER)
        for pos in positive
    ]
    df = pd.DataFrame(
        {
            "review_id": np.arange(1, n + 1),
            "date": dates.strftime("%Y-%m-%d"),
            "product": product,
            "region": region,
            "price": price.round(2),
            "delivery_days": rng.integers(1, 15, n).astype(float),
            "rating": rating,
            "review_text": text,
            "sentiment": np.where(positive, "positive", "negative"),
        }
    )
    df.loc[rng.choice(n, 40, replace=False), "delivery_days"] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "region"] = np.nan
    return df
