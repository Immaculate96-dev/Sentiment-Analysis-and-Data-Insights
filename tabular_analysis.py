"""
Tabular data analysis
---------------------
Column type detection, profiling, outliers, correlations, plain-English
insights and charts for any uploaded dataset. No Streamlit code in here, so
everything can be tested and reused on its own.

Column kinds:
    numeric, categorical, datetime, text, id, constant, empty
"""

from __future__ import annotations

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plotting import COLORS, apply_style, new_axes

apply_style()

KINDS = ["numeric", "categorical", "datetime", "text", "id", "constant", "empty"]
ANALYSABLE = ("numeric", "categorical", "datetime")

_ID_NAME = re.compile(r"(^|[_\s-])(id|index|key|uuid)$|^unnamed", re.IGNORECASE)


# ------------------------------------------------------- type detection
def _unique_names(names):
    seen, out = {}, []
    for name in names:
        name = name or "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _try_numeric(s: pd.Series):
    """Parse text such as '1,234', '$5.50' or '12%' as numbers. None if < 95% parse."""
    text = s.astype("string")
    cleaned = text.str.replace(",", "", regex=False).str.replace(r"[$%€£\s]", "", regex=True)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    n_values = s.notna().sum()
    if n_values and parsed.notna().sum() / n_values >= 0.95:
        return parsed.astype(float)
    return None


def _to_datetime(s: pd.Series, dayfirst: bool = False) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (TypeError, ValueError):
            out = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out


def _date_order(values: pd.Series):
    """Work out whether slash/dash dates are day-first or month-first.

    Returns 'day', 'month', 'ambiguous' (every value could be read both ways)
    or None (not this style of date, or inconsistent).
    """
    parts = values.astype(str).str.extract(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}")
    if parts.notna().all(axis=1).mean() < 0.5:
        return None
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    first_big, second_big = bool((first > 12).any()), bool((second > 12).any())
    if first_big and not second_big:
        return "day"
    if second_big and not first_big:
        return "month"
    if first_big and second_big:
        return None
    return "ambiguous"


def _parse_dates(s: pd.Series, dayfirst_default: bool = False):
    """Parse text as dates. Returns (parsed series, order note)."""
    values = s.dropna().astype(str)
    order = _date_order(values)
    if order == "day":
        dayfirst = True
    elif order == "month":
        dayfirst = False
    else:
        dayfirst = dayfirst_default
    note = order
    if order == "ambiguous":
        note = "ambiguous_day" if dayfirst else "ambiguous_month"
    return _to_datetime(s.astype("string"), dayfirst), note


def _try_datetime(s: pd.Series, dayfirst_default: bool = False):
    """Parse text as dates if the values look like dates. None otherwise."""
    values = s.dropna().astype(str)
    if values.empty:
        return None
    sample = values.sample(min(200, len(values)), random_state=0)
    if sample.str.len().mean() > 40 or sample.str.contains(r"\d").mean() < 0.9:
        return None
    if _to_datetime(sample, dayfirst_default).notna().mean() < 0.9:
        return None
    return _parse_dates(s, dayfirst_default)


def _classify_numeric(s: pd.Series, name: str) -> str:
    values = s.dropna()
    unique = values.nunique()
    if unique <= 1:
        return "constant"
    if unique == 2:
        return "categorical"  # e.g. 0/1 flags and labels
    is_whole = bool((values % 1 == 0).all())
    if len(values) >= 20 and is_whole and unique == len(values):
        is_counter = values.is_monotonic_increasing and bool((values.diff().dropna() == 1).all())
        if _ID_NAME.search(name) or is_counter:
            return "id"
    return "numeric"


def _classify_text(s: pd.Series) -> str:
    values = s.dropna().astype(str)
    sample = values.sample(min(2000, len(values)), random_state=0)
    avg_words = sample.str.split().str.len().mean()
    unique = values.nunique()
    ratio = unique / len(values)
    if unique <= 1:
        return "constant"
    if avg_words >= 6:
        return "text"
    if ratio > 0.9 and len(values) >= 20:
        return "id"
    if unique <= 200 or ratio <= 0.5:
        return "categorical"
    return "id"


def convert_column(s: pd.Series, kind: str, dayfirst: bool = False) -> pd.Series:
    """Force a column into a kind (used when the user overrides detection)."""
    if kind == "numeric":
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        parsed = _try_numeric(s)
        return parsed if parsed is not None else pd.to_numeric(s, errors="coerce")
    if kind == "datetime":
        return s if pd.api.types.is_datetime64_any_dtype(s) else _parse_dates(s, dayfirst)[0]
    if kind in ("categorical", "text", "id"):
        if pd.api.types.is_datetime64_any_dtype(s):
            return s
        return s.astype("string").where(s.notna())
    return s


def apply_overrides(df: pd.DataFrame, types: dict, overrides: dict, dayfirst: bool = False):
    """Apply user-chosen column kinds on top of the detected ones."""
    df, types = df.copy(), dict(types)
    for col, kind in overrides.items():
        if col in df.columns:
            df[col] = convert_column(df[col], kind, dayfirst)
            types[col] = kind
    return df, types


def _convert_and_classify(s: pd.Series, name: str, dayfirst: bool = False):
    """Returns (converted series, kind, date-order note or None)."""
    if s.notna().sum() == 0:
        return s, "empty", None
    if pd.api.types.is_bool_dtype(s):
        return s, "categorical", None
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s, "categorical", None
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, "datetime", None
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_float_dtype(s):
            s = s.replace([np.inf, -np.inf], np.nan)
        return s, _classify_numeric(s, name), None

    # text-like column: it might really hold numbers or dates
    text = s.astype("string").where(s.notna())
    parsed = _try_numeric(text)
    if parsed is not None:
        return parsed, _classify_numeric(parsed, name), None
    dates = _try_datetime(text, dayfirst)
    if dates is not None:
        return dates[0], "datetime", dates[1]
    return text, _classify_text(text), None


def prepare_dataframe(df: pd.DataFrame, dayfirst: bool = False):
    """Tidy column names, convert text that is really numbers/dates, and detect
    what each column holds.

    `dayfirst` only matters for dates that could be read either way, such as
    03/04/2024. Dates with a day above 12 are detected automatically.

    Returns (prepared DataFrame, {column name: kind}). The DataFrame's
    `attrs["date_order"]` records how each text date column was read.
    """
    df = df.copy()
    df.columns = _unique_names([str(c).strip() for c in df.columns])
    types, orders = {}, {}
    for col in df.columns:
        df[col], types[col], order = _convert_and_classify(df[col], col, dayfirst)
        if order:
            orders[col] = order
    df.attrs["date_order"] = orders
    return df, types


def columns_of(types: dict, *kinds) -> list:
    return [c for c, k in types.items() if k in kinds]


# -------------------------------------------------------------- profiling
def overview(df: pd.DataFrame, types: dict) -> dict:
    cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    kind_counts = pd.Series(types).value_counts().to_dict()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": missing,
        "missing_pct": (missing / cells * 100) if cells else 0.0,
        "duplicate_rows": int(df.duplicated().sum()) if len(df) else 0,
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "kind_counts": kind_counts,
    }


def column_profile(df: pd.DataFrame, types: dict) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        example = "" if non_null.empty else str(non_null.iloc[0])
        rows.append(
            {
                "column": col,
                "type": types[col],
                "missing": int(s.isna().sum()),
                "missing %": round(s.isna().mean() * 100, 1) if len(s) else 0.0,
                "unique": int(non_null.nunique()),
                "example": example[:60],
            }
        )
    return pd.DataFrame(rows)


def iqr_outliers(s: pd.Series):
    """Count values outside 1.5 x IQR. Returns (count, lower fence, upper fence)."""
    values = s.dropna()
    if values.empty:
        return 0, np.nan, np.nan
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0, q1, q3
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((values < lower) | (values > upper)).sum()), float(lower), float(upper)


def numeric_summary(df: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col].dropna()
        n_out, _, _ = iqr_outliers(s)
        rows.append(
            {
                "column": col,
                "count": int(s.count()),
                "missing": int(df[col].isna().sum()),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
                "skew": s.skew() if len(s) > 2 else np.nan,
                "outliers": n_out,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def correlation_matrix(df: pd.DataFrame, cols, method: str = "pearson") -> pd.DataFrame:
    cols = list(cols)[:40]  # keep it readable and fast
    return df[cols].corr(method=method, min_periods=3)


def top_correlations(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    pairs = []
    names = list(corr.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                pairs.append({"column A": a, "column B": b, "correlation": round(float(r), 3)})
    out = pd.DataFrame(pairs, columns=["column A", "column B", "correlation"])
    if out.empty:
        return out
    return out.reindex(out["correlation"].abs().sort_values(ascending=False).index).head(n).reset_index(drop=True)


def categorical_summary(s: pd.Series, top: int = 15) -> pd.DataFrame:
    counts = s.dropna().astype(str).value_counts()
    total = counts.sum()
    out = pd.DataFrame({"value": counts.index, "count": counts.values})
    out["percent"] = (out["count"] / total * 100).round(1) if total else 0.0
    return out.head(top)


def _fmt(x) -> str:
    if pd.isna(x):
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.3g}"


def generate_insights(df: pd.DataFrame, types: dict) -> list:
    """Plain-English findings. Each item is (level, text); level is
    'attention' (worth checking) or 'note' (good to know)."""
    out = []
    n = len(df)
    if n == 0:
        return [("attention", "The dataset has no rows.")]

    dupes = int(df.duplicated().sum())
    if dupes:
        out.append(("attention", f"{dupes:,} rows ({dupes / n:.1%}) are exact duplicates of another row."))

    for col in df.columns:
        miss = df[col].isna().mean()
        if types[col] == "empty":
            out.append(("attention", f"`{col}` is completely empty."))
        elif miss >= 0.2:
            out.append(("attention", f"`{col}` is {miss:.0%} missing."))
        elif miss > 0:
            out.append(("note", f"`{col}` has {miss:.1%} missing values."))

    constants = columns_of(types, "constant")
    if constants:
        out.append(("note", "These columns have the same value in every row: " + ", ".join(f"`{c}`" for c in constants) + "."))

    ids = columns_of(types, "id")
    if ids:
        out.append(("note", "These look like identifiers (almost every value is unique), so they are left out of the charts: "
                    + ", ".join(f"`{c}`" for c in ids) + "."))

    numeric = columns_of(types, "numeric")
    for col in numeric:
        s = df[col].dropna()
        if len(s) < 8:
            continue
        n_out, lo, hi = iqr_outliers(s)
        if n_out and n_out / len(s) >= 0.01:
            out.append(("attention", f"`{col}` has {n_out:,} outliers ({n_out / len(s):.1%}) outside {_fmt(lo)} to {_fmt(hi)}."))
        skew = s.skew()
        if abs(skew) > 1.5:
            if skew > 0:
                why = "a few very large values pull the mean above the median"
            else:
                why = "a few very small values pull the mean below the median"
            out.append(("note", f"`{col}` is strongly skewed (skewness {skew:.1f}): {why} (mean {_fmt(s.mean())}, median {_fmt(s.median())})."))

    if len(numeric) >= 2:
        corr = correlation_matrix(df, numeric)
        for _, row in top_correlations(corr, n=5).iterrows():
            r = row["correlation"]
            if abs(r) >= 0.7:
                direction = "rise together" if r > 0 else "move in opposite directions"
                out.append(("note", f"`{row['column A']}` and `{row['column B']}` are strongly correlated (r = {r:.2f}): they {direction}."))

    for col in columns_of(types, "categorical"):
        counts = df[col].dropna().astype(str).value_counts(normalize=True)
        if len(counts) >= 3 and counts.iloc[0] >= 0.8:
            out.append(("note", f"`{col}` is dominated by '{counts.index[0]}' ({counts.iloc[0]:.0%} of rows)."))

    orders = df.attrs.get("date_order", {})
    for col in columns_of(types, "datetime"):
        s = df[col].dropna()
        if not s.empty:
            out.append(("note", f"`{col}` runs from {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d}."))
        if orders.get(col) == "ambiguous_month":
            out.append(("attention", f"Dates in `{col}` can be read two ways (03/04/2024 could be 3 April or 4 March). "
                        "They were read as month/day. If yours are day/month, switch the option in the sidebar."))
        elif orders.get(col) == "ambiguous_day":
            out.append(("note", f"Dates in `{col}` were read as day/month, as you chose in the sidebar."))

    texts = columns_of(types, "text")
    if texts:
        out.append(("note", "Free-text columns found: " + ", ".join(f"`{c}`" for c in texts) + ". Try the Text sentiment tab."))

    return out


# ----------------------------------------------------------------- charts
def plot_missing(df: pd.DataFrame, ax=None):
    fig, ax = new_axes(ax, figsize=(6, max(2.5, 0.35 * df.shape[1] + 1)))
    pct = (df.isna().mean() * 100).sort_values()
    pct = pct[pct > 0]
    if pct.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    ax.barh(pct.index.astype(str), pct.values, color=COLORS["negative"])
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    ax.grid(axis="y", visible=False)
    return fig


def plot_numeric_distribution(s: pd.Series, name: str):
    values = s.dropna()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 4.2), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    ax1.hist(values, bins=min(40, max(5, int(np.sqrt(len(values))))), color=COLORS["accent"])
    ax1.axvline(values.median(), color=COLORS["text"], linestyle="--", linewidth=1, label=f"median {_fmt(values.median())}")
    ax1.legend(frameon=False)
    ax1.set_title(f"Distribution of {name}")
    ax1.set_ylabel("Rows")
    ax2.boxplot(values, vert=False, widths=0.6, patch_artist=True,
                boxprops={"facecolor": COLORS["accent_light"], "edgecolor": COLORS["accent"]},
                medianprops={"color": COLORS["text"]}, flierprops={"markersize": 3, "markeredgecolor": COLORS["negative"]})
    ax2.set_yticks([])
    ax2.grid(False)
    ax2.set_xlabel(name)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame):
    k = len(corr)
    size = min(9, max(4, 0.6 * k + 2))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.grid(False)
    if k <= 12:
        for i in range(k):
            for j in range(k):
                v = corr.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.6 else COLORS["text"])
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation between numeric columns")
    fig.tight_layout()
    return fig


def plot_category_bars(summary: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.35 * len(summary) + 1)))
    labels = summary["value"].astype(str).str.slice(0, 30)[::-1]
    ax.barh(labels, summary["count"][::-1], color=COLORS["accent"])
    for y, (cnt, pct) in enumerate(zip(summary["count"][::-1], summary["percent"][::-1])):
        ax.text(cnt, y, f"  {pct:.1f}%", va="center", fontsize=9)
    ax.set_title(f"Most common values in {name}")
    ax.set_xlabel("Rows")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def _period_for(span_days: float) -> str:
    if span_days <= 31:
        return "D"
    if span_days <= 180:
        return "W"
    if span_days <= 1500:
        return "M"
    if span_days <= 3650:
        return "Q"
    return "Y"


_PERIOD_NAMES = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}


def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str = None, agg: str = "mean", ax=None):
    """Rows per period (value_col=None) or mean/sum of a numeric column per period."""
    data = df[[date_col] + ([value_col] if value_col else [])].dropna(subset=[date_col])
    fig, ax = new_axes(ax, figsize=(7, 3.6))
    if data.empty:
        ax.text(0.5, 0.5, "No valid dates", ha="center", va="center", transform=ax.transAxes)
        return fig
    span = (data[date_col].max() - data[date_col].min()).days
    freq = _period_for(span)
    period = data[date_col].dt.to_period(freq)
    if value_col is None:
        series = data.groupby(period).size()
        ylabel = "Rows"
    else:
        series = data.groupby(period)[value_col].agg(agg)
        ylabel = f"{agg} of {value_col}"
    series.index = series.index.to_timestamp()
    ax.plot(series.index, series.values, color=COLORS["accent"], linewidth=2, marker="o", markersize=3)
    ax.set_title(f"{ylabel} per {_PERIOD_NAMES[freq]}")
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    return fig


def plot_group_means(df: pd.DataFrame, cat_col: str, value_col: str, top: int = 12):
    """Average of value_col for the most common values of cat_col."""
    data = df[[cat_col, value_col]].dropna().copy()
    data[cat_col] = data[cat_col].astype(str)
    top_cats = data[cat_col].value_counts().head(top).index
    grouped = data[data[cat_col].isin(top_cats)].groupby(cat_col)[value_col].agg(["mean", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.4 * len(grouped) + 1)))
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in grouped["mean"]]
    ax.barh(grouped.index.str.slice(0, 30), grouped["mean"], color=colors)
    ax.axvline(0, color=COLORS["text"], linewidth=0.8)
    ax.set_title(f"Average {value_col} by {cat_col}")
    ax.set_xlabel(f"Average {value_col}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------- relationships
def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _strength(r: float) -> str:
    a = abs(r)
    word = "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else "strong" if a < 0.8 else "very strong"
    return word


def relationship(df: pd.DataFrame, x: str, y: str, types: dict):
    """Pick a sensible chart and statistic for any pair of columns.

    Returns (figure or None, explanation text).
    """
    if x == y:
        return None, "Pick two different columns."
    kx, ky = types[x], types[y]
    if kx not in ANALYSABLE or ky not in ANALYSABLE:
        return None, "Relationships work for numeric, category and date columns. Pick different columns."
    if kx == "datetime" and ky == "datetime":
        return None, "Pick at most one date column."
    # put the date or the category first so each case is handled once
    if ky == "datetime" or (ky == "categorical" and kx == "numeric"):
        x, y, kx, ky = y, x, ky, kx

    if kx == "numeric" and ky == "numeric":
        return _rel_num_num(df, x, y)
    if kx == "categorical" and ky == "numeric":
        return _rel_cat_num(df, x, y)
    if kx == "categorical" and ky == "categorical":
        return _rel_cat_cat(df, x, y)
    if kx == "datetime" and ky == "numeric":
        fig = plot_time_series(df, x, y, "mean")
        return fig, f"Average of `{y}` over time. Use this to spot trends or seasonality."
    if kx == "datetime" and ky == "categorical":
        return _rel_date_cat(df, x, y)
    return None, "That combination isn't supported."


def _rel_num_num(df, x, y):
    data = df[[x, y]].dropna()
    if len(data) < 3:
        return None, "Not enough rows with both values to compare."
    pearson, p_pearson = stats.pearsonr(data[x], data[y]) if data[x].nunique() > 1 and data[y].nunique() > 1 else (np.nan, np.nan)
    spearman, _ = stats.spearmanr(data[x], data[y])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plot = data.sample(5000, random_state=0) if len(data) > 5000 else data
    ax.scatter(plot[x], plot[y], s=14, alpha=0.45, color=COLORS["accent"], edgecolor="none")
    if data[x].nunique() > 1:
        slope, intercept = np.polyfit(data[x], data[y], 1)
        xs = np.array([data[x].min(), data[x].max()])
        ax.plot(xs, slope * xs + intercept, color=COLORS["negative"], linewidth=2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    fig.tight_layout()
    if np.isnan(pearson):
        return fig, f"Based on {len(data):,} rows. One of the columns doesn't vary, so no correlation can be computed."
    direction = "positive" if pearson > 0 else "negative"
    text = (f"Pearson r = {pearson:.2f} ({_p_text(p_pearson)}), Spearman rho = {spearman:.2f}, based on {len(data):,} rows. "
            f"That is a {_strength(pearson)} {direction} relationship. Correlation alone doesn't show that one column causes the other.")
    return fig, text


def _rel_cat_num(df, cat, num):
    data = df[[cat, num]].dropna().copy()
    data[cat] = data[cat].astype(str)
    top = data[cat].value_counts().head(10).index
    data = data[data[cat].isin(top)]
    groups = [(name, g[num].values) for name, g in data.groupby(cat) if len(g) >= 1]
    groups.sort(key=lambda t: np.median(t[1]))
    if len(groups) < 2:
        return None, "Need at least two categories with data."
    fig, ax = plt.subplots(figsize=(max(5.5, 0.8 * len(groups) + 2), 4.2))
    ax.boxplot([g for _, g in groups], patch_artist=True,
               boxprops={"facecolor": COLORS["accent_light"], "edgecolor": COLORS["accent"]},
               medianprops={"color": COLORS["text"]}, flierprops={"markersize": 3})
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([n[:18] for n, _ in groups], rotation=30, ha="right")
    ax.set_ylabel(num)
    ax.set_title(f"{num} by {cat}")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    medians = ", ".join(f"{n}: {_fmt(np.median(g))}" for n, g in groups[::-1][:5])
    text = f"Median of `{num}` per group (highest first): {medians}."
    testable = [g for _, g in groups if len(g) >= 2]
    if len(testable) >= 2 and not all(len(set(g)) == 1 for g in testable) and len(np.unique(np.concatenate(testable))) > 1:
        _, p = stats.kruskal(*testable)
        verdict = "The groups differ more than chance would explain." if p < 0.05 else "The differences between groups could plausibly be chance."
        text += f" Kruskal-Wallis test: {_p_text(p)}. {verdict}"
    return fig, text


def _rel_cat_cat(df, a, b):
    data = df[[a, b]].dropna().astype(str)
    top_a = data[a].value_counts().head(10).index
    top_b = data[b].value_counts().head(10).index
    data = data[data[a].isin(top_a) & data[b].isin(top_b)]
    table = pd.crosstab(data[a], data[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, "Both columns need at least two different values."
    share = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * table.shape[1] + 3), max(3.5, 0.5 * table.shape[0] + 2)))
    im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_yticks(range(table.shape[0]))
    ax.set_xticklabels([c[:18] for c in table.columns], rotation=30, ha="right")
    ax.set_yticklabels([r[:22] for r in table.index])
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{share.values[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if share.values[i, j] > 0.6 else COLORS["text"])
    ax.set_xlabel(b)
    ax.set_ylabel(a)
    ax.set_title(f"Share of each {b} within {a}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else np.nan
    verdict = "The two columns appear related." if p < 0.05 else "No clear relationship."
    return fig, f"Each row sums to 100%. Chi-square test: {_p_text(p)}; Cramer's V = {v:.2f} (0 = unrelated, 1 = perfectly related). {verdict}"


def _rel_date_cat(df, date_col, cat):
    data = df[[date_col, cat]].dropna().copy()
    data[cat] = data[cat].astype(str)
    top = data[cat].value_counts().head(5).index
    data = data[data[cat].isin(top)]
    if data.empty:
        return None, "No rows with both values."
    span = (data[date_col].max() - data[date_col].min()).days
    freq = _period_for(span)
    data = data.assign(period=data[date_col].dt.to_period(freq))
    counts = data.groupby(["period", cat]).size().unstack(fill_value=0)
    counts.index = counts.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for col in counts.columns:
        ax.plot(counts.index, counts[col], linewidth=2, label=str(col)[:20])
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Rows")
    ax.set_title(f"Rows per {_PERIOD_NAMES[freq]} by {cat}")
    fig.autofmt_xdate()
    return fig, f"Row counts per {_PERIOD_NAMES[freq]} for the five most common values of `{cat}`."
