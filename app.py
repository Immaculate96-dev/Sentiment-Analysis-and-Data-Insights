"""
Dataset analyser - web app
--------------------------
Upload a CSV or Excel file and get an automatic overview, column-by-column
analysis, relationships between columns and, for text columns, sentiment
analysis.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import csv
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import sentiment_analysis as sa  # noqa: E402
import tabular_analysis as ta  # noqa: E402
from demo_data import make_demo_dataframe  # noqa: E402

MAX_ROWS = 250_000  # larger files are sampled so the app stays responsive
SENTIMENT_DEFAULT_ROWS = 10_000
SENTIMENT_MAX_ROWS = 50_000

st.set_page_config(page_title="Dataset analyser", layout="wide")


# ------------------------------------------------------------------ helpers
def show(fig):
    """Render a matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def _detect_separator(sample: str, filename: str) -> str:
    if filename.endswith(".tsv"):
        return "\t"
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


@st.cache_data(show_spinner="Reading file...")
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(file_bytes))
    if name.endswith(".json"):
        return pd.read_json(io.BytesIO(file_bytes))
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            sample = file_bytes[:20000].decode(encoding, errors="ignore")
            sep = _detect_separator(sample, name)
            return pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file. Save it as UTF-8 CSV and try again.")


@st.cache_data(show_spinner="Detecting column types...")
def prepare(df: pd.DataFrame, dayfirst: bool = False):
    return ta.prepare_dataframe(df, dayfirst)


@st.cache_data(show_spinner="Scoring text with VADER...")
def run_vader(texts: pd.Series) -> pd.DataFrame:
    return sa.score_vader(texts)


@st.cache_data(show_spinner="Training the model...")
def run_training(texts: pd.Series, labels: pd.Series) -> dict:
    return sa.train_classifier(texts, labels)


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    return make_demo_dataframe()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------- load a dataset
st.title("Dataset analyser")

with st.sidebar:
    st.header("Your data")
    uploaded = st.file_uploader("Upload a CSV, TSV, Excel or JSON file", type=["csv", "tsv", "txt", "xlsx", "xlsm", "json"])
    if st.button("Try sample data"):
        st.session_state["use_demo"] = True
    if uploaded is not None:
        st.session_state["use_demo"] = False
    st.caption("Files are analysed in memory and are not stored by the app.")

if uploaded is not None:
    source_name = uploaded.name
    try:
        raw = load_file(uploaded.getvalue(), uploaded.name)
    except Exception as err:  # unreadable file: tell the user what to fix
        st.error(f"Could not read this file: {err}")
        st.stop()
elif st.session_state.get("use_demo"):
    source_name = "sample data (product reviews)"
    raw = demo_data()
else:
    st.write("Upload a dataset in the sidebar to get an overview, charts for every column, "
             "relationships between columns and sentiment analysis for text.")
    st.write("No file handy? Choose **Try sample data** in the sidebar.")
    st.stop()

if raw.empty or raw.shape[1] == 0:
    st.error("This file has no data. Check that it has a header row and at least one row of values.")
    st.stop()

if len(raw) > MAX_ROWS:
    st.warning(f"This file has {len(raw):,} rows. To keep the app responsive, a random sample of {MAX_ROWS:,} rows is analysed.")
    raw = raw.sample(MAX_ROWS, random_state=42).reset_index(drop=True)

df, detected_types = prepare(raw, False)
dayfirst = False
if any(o.startswith("ambiguous") for o in df.attrs.get("date_order", {}).values()):
    with st.sidebar:
        dayfirst = st.checkbox("Read dates like 03/04/2024 as day/month", value=False,
                               help="Some dates can be read two ways. Tick this if yours put the day first.")
    if dayfirst:
        df, detected_types = prepare(raw, True)

# user overrides of column types, kept per dataset
dataset_key = f"{source_name}|{raw.shape}"
if st.session_state.get("override_key") != dataset_key:
    st.session_state["override_key"] = dataset_key
    st.session_state["overrides"] = {}
    st.session_state.pop("sent_result", None)
df, types = ta.apply_overrides(df, detected_types, st.session_state["overrides"], dayfirst)

with st.sidebar:
    st.divider()
    st.write(f"**{source_name}**")
    st.write(f"{len(df):,} rows x {df.shape[1]} columns")

tabs = st.tabs(["Overview", "Columns", "Numbers", "Categories", "Dates", "Relationships", "Text sentiment", "Export"])
tab_overview, tab_columns, tab_numbers, tab_categories, tab_dates, tab_rel, tab_text, tab_export = tabs

# ----------------------------------------------------------------- overview
with tab_overview:
    ov = ta.overview(df, types)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ov['rows']:,}")
    c2.metric("Columns", f"{ov['columns']}")
    c3.metric("Missing cells", f"{ov['missing_pct']:.1f}%")
    c4.metric("Duplicate rows", f"{ov['duplicate_rows']:,}")
    st.caption("Column types: " + ", ".join(f"{n} {k}" for k, n in ov["kind_counts"].items()))

    insights = ta.generate_insights(df, types)
    attention = [t for lvl, t in insights if lvl == "attention"]
    notes = [t for lvl, t in insights if lvl == "note"]
    if attention:
        st.subheader("Worth checking")
        for text in attention:
            st.markdown(f"- {text}")
    if notes:
        st.subheader("Good to know")
        for text in notes:
            st.markdown(f"- {text}")
    st.subheader("First rows")
    st.dataframe(df.head(50))

# ------------------------------------------------------------------ columns
with tab_columns:
    st.subheader("Every column")
    st.dataframe(ta.column_profile(df, types))
    show(ta.plot_missing(df))

    with st.expander("A column looks wrong? Change how it is treated"):
        col_pick = st.selectbox("Column", list(df.columns), key="override_col")
        kind_pick = st.selectbox("Treat as", ta.KINDS[:5], index=ta.KINDS.index(types[col_pick]) if types[col_pick] in ta.KINDS[:5] else 0,
                                 key="override_kind")
        if st.button("Apply"):
            st.session_state["overrides"][col_pick] = kind_pick
            st.rerun()
        if st.session_state["overrides"]:
            st.caption("Changed: " + ", ".join(f"{c} -> {k}" for c, k in st.session_state["overrides"].items()))
            if st.button("Reset all"):
                st.session_state["overrides"] = {}
                st.rerun()

# ------------------------------------------------------------------ numbers
with tab_numbers:
    numeric_cols = ta.columns_of(types, "numeric")
    if not numeric_cols:
        st.info("No numeric columns were found. If a column holds numbers, set it to numeric in the Columns tab.")
    else:
        st.subheader("Summary")
        st.dataframe(ta.numeric_summary(df, numeric_cols).round(3))
        col = st.selectbox("Look at one column", numeric_cols, key="num_col")
        n_out, lo, hi = ta.iqr_outliers(df[col])
        left, right = st.columns([2, 1])
        with left:
            show(ta.plot_numeric_distribution(df[col], col))
        with right:
            st.metric("Outliers", f"{n_out:,}")
            if n_out:
                st.caption(f"Values below {lo:,.4g} or above {hi:,.4g}.")
                outlier_rows = df[(df[col] < lo) | (df[col] > hi)]
                with st.expander("Show outlier rows"):
                    st.dataframe(outlier_rows.head(200))
        if len(numeric_cols) >= 2:
            st.subheader("Correlations")
            method = st.radio("Method", ["pearson", "spearman"], horizontal=True,
                              help="Pearson finds straight-line relationships. Spearman also finds curved ones and resists outliers.")
            corr = ta.correlation_matrix(df, numeric_cols, method)
            show(ta.plot_correlation_heatmap(corr))
            st.dataframe(ta.top_correlations(corr, 10))

# --------------------------------------------------------------- categories
with tab_categories:
    cat_cols = ta.columns_of(types, "categorical")
    if not cat_cols:
        st.info("No category columns were found.")
    else:
        col = st.selectbox("Column", cat_cols, key="cat_col")
        top_n = st.slider("Values to show", 5, 30, 12)
        summary = ta.categorical_summary(df[col], top_n)
        left, right = st.columns([3, 2])
        with left:
            show(ta.plot_category_bars(summary, col))
        with right:
            st.dataframe(summary)
            st.caption(f"{df[col].nunique():,} different values, {df[col].isna().sum():,} missing.")

# -------------------------------------------------------------------- dates
with tab_dates:
    date_cols = ta.columns_of(types, "datetime")
    if not date_cols:
        st.info("No date columns were found. If a column holds dates, set it to datetime in the Columns tab.")
    else:
        date_col = st.selectbox("Date column", date_cols, key="date_col")
        numeric_cols = ta.columns_of(types, "numeric")
        measure = st.selectbox("Show", ["Number of rows"] + numeric_cols, key="date_measure")
        agg = "mean"
        if measure != "Number of rows":
            agg = st.radio("Combine values by", ["mean", "sum"], horizontal=True)
        dates = df[date_col].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("From", f"{dates.min():%Y-%m-%d}" if len(dates) else "-")
        c2.metric("To", f"{dates.max():%Y-%m-%d}" if len(dates) else "-")
        c3.metric("Rows without a date", f"{df[date_col].isna().sum():,}")
        show(ta.plot_time_series(df, date_col, None if measure == "Number of rows" else measure, agg))

# ------------------------------------------------------------ relationships
with tab_rel:
    usable = [c for c in df.columns if types[c] in ta.ANALYSABLE]
    if len(usable) < 2:
        st.info("Relationships need at least two numeric, category or date columns.")
    else:
        c1, c2 = st.columns(2)
        x = c1.selectbox("First column", usable, index=0, key="rel_x")
        y = c2.selectbox("Second column", usable, index=1, key="rel_y")
        fig, explanation = ta.relationship(df, x, y, types)
        if fig is not None:
            show(fig)
        st.write(explanation)

# ----------------------------------------------------------- text sentiment
with tab_text:
    text_candidates = ta.columns_of(types, "text") + [
        c for c in ta.columns_of(types, "categorical", "id") if c not in ta.columns_of(types, "text")
    ]
    if not text_candidates:
        st.info("No text columns were found. Sentiment analysis needs a column of sentences, such as reviews or comments.")
    else:
        st.write("Score each row of a text column with VADER. If your data also has a label column "
                 "(for example positive/negative), a model is trained on it and compared with VADER.")
        text_col = st.selectbox("Text column", text_candidates, key="sent_text")
        label_options = ["None (no labels)"] + [c for c in ta.columns_of(types, "categorical") if c != text_col]
        default_label = next((i for i, c in enumerate(label_options) if "sentiment" in c.lower() or "label" in c.lower()), 0)
        label_pick = st.selectbox("Label column (optional)", label_options, index=default_label, key="sent_label")
        label_col = None if label_pick == label_options[0] else label_pick

        positive_value = None
        if label_col is not None:
            values = sorted(df[label_col].dropna().astype(str).unique())
            if len(values) == 2:
                guess = sa.guess_positive_value(values)
                positive_value = st.selectbox("Which value means positive?", values, index=values.index(guess), key="sent_pos")
            elif len(values) > 2:
                st.caption(f"{len(values)} label values found. A model will be trained on all of them; "
                           "VADER can't be compared because it only separates positive from negative.")

        n_available = int(df[text_col].notna().sum())
        if n_available == 0:
            st.warning("This column has no text.")
        else:
            cap = min(n_available, SENTIMENT_MAX_ROWS)
            default_rows = min(cap, SENTIMENT_DEFAULT_ROWS)
            if cap > 200:
                n_rows = st.slider("Rows to analyse", 100, cap, default_rows, step=100,
                                   help="More rows take longer. A random sample is used.")
            else:
                n_rows = cap
            params = (source_name, text_col, label_col, positive_value, n_rows)

            if st.button("Run sentiment analysis", type="primary"):
                work_cols = [text_col] + ([label_col] if label_col else [])
                work = df[work_cols].dropna(subset=[text_col])
                if len(work) > n_rows:
                    work = work.sample(n_rows, random_state=42)
                texts = sa.clean_series(work[text_col])
                keep = texts.str.len() > 0
                work, texts = work[keep], texts[keep]
                result = {"params": params, "index": texts.index.tolist(), "texts": texts}
                result["vader"] = run_vader(texts)
                if label_col is not None:
                    labels = work[label_col]
                    if positive_value is not None:
                        labels = sa.to_binary_sentiment(labels, positive_value)
                    try:
                        result["model"] = run_training(texts, labels)
                        result["labels"] = labels
                    except ValueError as err:
                        result["model_error"] = str(err)
                st.session_state["sent_result"] = result

            result = st.session_state.get("sent_result")
            if result and result["params"] != params:
                st.info("Settings changed. Click **Run sentiment analysis** to update the results.")
            elif result:
                vader = result["vader"]
                model = result.get("model")
                texts = result["texts"]

                st.subheader("Results")
                metrics = st.columns(4)
                metrics[0].metric("Rows analysed", f"{len(texts):,}")
                pos_share = (vader["vader_label"] == "positive").mean()
                neg_share = (vader["vader_label"] == "negative").mean()
                metrics[1].metric("Positive (VADER)", f"{pos_share:.0%}")
                metrics[2].metric("Negative (VADER)", f"{neg_share:.0%}")
                metrics[3].metric("Average score", f"{vader['vader_score'].mean():+.2f}")

                left, right = st.columns(2)
                with left:
                    show(sa.plot_label_distribution(vader["vader_label"].value_counts(), "Sentiment according to VADER"))
                with right:
                    show(sa.plot_vader_hist(vader["vader_score"]))

                if "model_error" in result:
                    st.warning(result["model_error"])

                if model is not None:
                    st.subheader("Model compared with VADER")
                    accuracies = {}
                    if positive_value is not None:
                        test_idx = model["y_test"].index
                        true_test = result["labels"].loc[test_idx]
                        accuracies["VADER"] = float((vader.loc[test_idx, "vader_binary"] == true_test).mean())
                    accuracies["Always the most common label"] = model["majority_accuracy"]
                    if model["baseline_accuracy"] is not None:
                        accuracies["Model (single words)"] = model["baseline_accuracy"]
                    accuracies["Model (words + pairs)"] = model["accuracy"]
                    m1, m2 = st.columns([3, 2])
                    with m1:
                        show(sa.plot_accuracy_compare(accuracies))
                    with m2:
                        st.write(f"All scores are measured on the same **{model['n_test']:,}** held-out rows "
                                 f"that the model never saw during training ({model['n_train']:,} rows were used to train it).")
                        st.write(f"Macro F1 of the model: **{model['macro_f1']:.2f}**")
                        st.caption("If the model barely beats 'always the most common label', the labels may be too noisy "
                                   "or the dataset too small to learn from.")

                    st.subheader("Which words matter most")
                    cols = st.columns(len(model["top_words"]) if len(model["top_words"]) <= 3 else 2)
                    for i, (label, words) in enumerate(model["top_words"].items()):
                        with cols[i % len(cols)]:
                            show(sa.plot_top_words(words, label))

                    left, right = st.columns(2)
                    with left:
                        show(sa.plot_confusion(model["confusion"], model["classes"]))
                    with right:
                        st.write("**Where the model is most confidently wrong**")
                        st.dataframe(model["errors"][["text", "actual", "predicted", "confidence"]].head(15))

                st.subheader("Break it down")
                group_cols = [c for c in ta.columns_of(types, "categorical") if 2 < df[c].nunique() <= 50 and c != label_col]
                date_cols = ta.columns_of(types, "datetime")
                scored = df.loc[result["index"]].assign(vader_score=vader["vader_score"])
                if group_cols:
                    group_pick = st.selectbox("Average VADER score by", group_cols, key="sent_group")
                    show(ta.plot_group_means(scored, group_pick, "vader_score"))
                if date_cols:
                    date_pick = st.selectbox("Average VADER score over time", date_cols, key="sent_date")
                    show(ta.plot_time_series(scored, date_pick, "vader_score", "mean"))
                if not group_cols and not date_cols:
                    st.caption("Add a category or date column to your data to see sentiment broken down.")

                st.subheader("Most and least positive rows")
                ranked = pd.DataFrame({"text": texts, "vader_score": vader["vader_score"]}).sort_values("vader_score")
                a, b = st.columns(2)
                a.write("Most negative")
                a.dataframe(ranked.head(10))
                b.write("Most positive")
                b.dataframe(ranked.tail(10).iloc[::-1])

                st.subheader("Try your own text")
                sample_text = st.text_area("Type or paste a sentence", key="try_text")
                if sample_text.strip():
                    score = run_vader(pd.Series([sa.clean_text(sample_text)]))["vader_score"].iloc[0]
                    st.write(f"VADER score: **{score:+.2f}** ({'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral'})")
                    if model is not None:
                        pred = sa.predict_text(model, sample_text)
                        probs = ", ".join(f"{k}: {v:.0%}" for k, v in pred["probabilities"].items())
                        st.write(f"Model prediction: **{pred['label']}** ({probs})")

# ------------------------------------------------------------------- export
with tab_export:
    st.subheader("Download")
    st.download_button("Cleaned data (CSV)", csv_bytes(df), "cleaned_data.csv", "text/csv")
    st.download_button("Column summary (CSV)", csv_bytes(ta.column_profile(df, types)), "column_summary.csv", "text/csv")
    result = st.session_state.get("sent_result")
    if result:
        out = pd.DataFrame({"text": result["texts"]}).join(result["vader"])
        st.download_button("Sentiment results (CSV)", csv_bytes(out), "sentiment_results.csv", "text/csv")
