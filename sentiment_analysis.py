"""
Sentiment analysis toolkit
--------------------------
Reusable functions for scoring text with VADER and training a TF-IDF +
logistic regression classifier. Used by the web app (app.py) and by the
command-line script at the bottom of this file.

Command-line use (original IMDB project):
    pip install -r requirements.txt
    # put "IMDB Dataset.csv" (Kaggle) in this folder, then:
    python sentiment_analysis.py
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from plotting import COLORS, apply_style, color_for_label, new_axes

# Standard stop-word lists remove "not", "no", "never", which would destroy the
# very thing word pairs are meant to capture, so the negation words are kept.
NEGATIONS = {"not", "no", "nor", "never", "cannot", "nothing", "nobody", "neither"}
CUSTOM_STOPS = sorted(ENGLISH_STOP_WORDS - NEGATIONS)

# Label values that usually mean "positive" (compared in lower case).
POSITIVE_VALUES = {"positive", "pos", "1", "1.0", "true", "yes", "good", "y", "+", "p"}

MAX_LABEL_CLASSES = 30


# ------------------------------------------------------------------ cleaning
def clean_text(text) -> str:
    """Remove HTML tags and extra whitespace. Case and punctuation are kept
    because VADER relies on them."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_series(texts: pd.Series) -> pd.Series:
    return texts.map(clean_text)


# --------------------------------------------------------------------- VADER
_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_vader(texts: pd.Series, neutral_band: float = 0.05) -> pd.DataFrame:
    """Score each text with VADER.

    Returns a DataFrame (same index as `texts`) with:
      vader_score  compound score from -1 (very negative) to +1 (very positive)
      vader_label  positive / neutral / negative, using the standard +-0.05 band
      vader_binary positive (score >= 0) or negative, as in the original project,
                   used when comparing against two-class labels
    """
    analyzer = _get_analyzer()
    scores = texts.map(lambda t: analyzer.polarity_scores(t)["compound"]).astype(float)
    label = np.select(
        [scores >= neutral_band, scores <= -neutral_band],
        ["positive", "negative"],
        default="neutral",
    )
    return pd.DataFrame(
        {
            "vader_score": scores,
            "vader_label": label,
            "vader_binary": np.where(scores >= 0, "positive", "negative"),
        },
        index=texts.index,
    )


# ------------------------------------------------------------------- labels
def guess_positive_value(values) -> str:
    """Pick the value in a two-class label column that most likely means positive."""
    values = [str(v) for v in values]
    for v in values:
        if v.strip().lower() in POSITIVE_VALUES:
            return v
    return sorted(values)[-1]


def to_binary_sentiment(labels: pd.Series, positive_value) -> pd.Series:
    """Map a two-class label column to 'positive' / 'negative'."""
    as_text = labels.astype(str)
    return pd.Series(
        np.where(as_text == str(positive_value), "positive", "negative"),
        index=labels.index,
    )


# ------------------------------------------------------------ ML classifier
def train_classifier(
    texts: pd.Series,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    compare_baseline: bool = True,
    stratify: bool = True,
    n_top: int = 12,
    n_errors: int = 25,
) -> dict:
    """Train TF-IDF (words + word pairs) + logistic regression on labelled text.

    Works for two or more classes. Returns a dictionary with the fitted model
    and vectorizer, accuracy on a held-out test set, a single-words baseline,
    the confusion matrix, the most predictive words per class and the test
    reviews the model got wrong.

    Raises ValueError with a plain-English message if the data cannot be used.
    """
    data = pd.DataFrame({"text": texts, "label": labels}).dropna()
    data["label"] = data["label"].astype(str)
    data = data[data["text"].astype(str).str.strip() != ""]

    counts = data["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("The label column needs at least two different values to train a model.")
    if len(counts) > MAX_LABEL_CLASSES:
        raise ValueError(
            f"The label column has {len(counts)} different values. "
            "Pick a column with a small number of categories, such as positive/negative."
        )
    if len(data) < 20 or counts.min() < 2:
        raise ValueError(
            "There are too few labelled rows to train a model. "
            "Each label needs at least 2 rows and the dataset at least 20."
        )

    split_args = dict(test_size=test_size, random_state=random_state)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data["text"], data["label"],
            stratify=data["label"] if stratify else None, **split_args,
        )
    except ValueError:  # too few rows per class for a stratified split
        X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], **split_args)

    # Model A: single words only (baseline)
    baseline_acc = None
    if compare_baseline:
        vec_uni = TfidfVectorizer(stop_words="english", max_features=20000)
        try:
            model_uni = LogisticRegression(max_iter=1000)
            model_uni.fit(vec_uni.fit_transform(X_train), y_train)
            baseline_acc = float((model_uni.predict(vec_uni.transform(X_test)) == y_test).mean())
        except ValueError:
            baseline_acc = None

    # Model B: words + two-word pairs
    vectorizer = TfidfVectorizer(
        stop_words=CUSTOM_STOPS,
        ngram_range=(1, 2),
        min_df=3 if len(X_train) >= 200 else 1,
        max_features=50000,
    )
    try:
        X_train_vec = vectorizer.fit_transform(X_train)
    except ValueError:
        raise ValueError("The text column has no usable words after cleaning. Pick a column with real sentences.")
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    confidence = model.predict_proba(X_test_vec).max(axis=1)
    classes = [str(c) for c in model.classes_]
    accuracy = float((pred == y_test.values).mean())

    # Most predictive words per class
    names = vectorizer.get_feature_names_out()
    coef = model.coef_
    top_words = {}
    if len(classes) == 2:
        w = coef[0]  # positive weight pushes towards classes[1]
        order = np.argsort(w)
        top_words[classes[1]] = [(names[i], float(w[i])) for i in order[::-1][:n_top]]
        top_words[classes[0]] = [(names[i], float(-w[i])) for i in order[:n_top]]
    else:
        for k, cls in enumerate(classes):
            order = np.argsort(coef[k])[::-1][:n_top]
            top_words[cls] = [(names[i], float(coef[k][i])) for i in order]

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    wrong = pd.DataFrame(
        {"text": X_test.values, "actual": y_test.values, "predicted": pred, "confidence": confidence},
        index=y_test.index,
    )
    wrong = wrong[wrong["actual"] != wrong["predicted"]].sort_values("confidence", ascending=False)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "classes": classes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_acc,
        "majority_accuracy": float(y_test.value_counts(normalize=True).max()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "report": report,
        "confusion": confusion_matrix(y_test, pred, labels=classes),
        "top_words": top_words,
        "errors": wrong.head(n_errors),
        "y_test": y_test,
        "pred": pred,
    }


def predict_text(result: dict, text: str) -> dict:
    """Classify one piece of text with a trained model from train_classifier()."""
    vec = result["vectorizer"].transform([clean_text(text)])
    proba = result["model"].predict_proba(vec)[0]
    classes = result["classes"]
    best = int(np.argmax(proba))
    return {"label": classes[best], "probabilities": dict(zip(classes, map(float, proba)))}


def frequent_words(texts, n: int = 10):
    """Most common meaningful words (stop words and film filler removed)."""
    extra = {"movie", "film", "br", "one", "like", "just", "really", "it's"}
    words = re.findall(r"[a-z']+", " ".join(map(str, texts)).lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and w not in extra and len(w) > 2]
    return Counter(words).most_common(n)


# -------------------------------------------------------------------- charts
def plot_label_distribution(counts: pd.Series, title="Sentiment distribution", ax=None):
    fig, ax = new_axes(ax)
    colors = [color_for_label(c) for c in counts.index]
    ax.bar([str(c) for c in counts.index], counts.values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="x", visible=False)
    return fig


def plot_vader_hist(scores: pd.Series, ax=None):
    fig, ax = new_axes(ax)
    ax.hist(scores, bins=40, color=COLORS["accent"])
    ax.axvline(0, color=COLORS["text"], linewidth=0.8, linestyle="--")
    ax.set_title("Distribution of VADER scores")
    ax.set_xlabel("Compound score (-1 very negative, +1 very positive)")
    ax.set_ylabel("Rows")
    return fig


def plot_accuracy_compare(accuracies: dict, ax=None):
    fig, ax = new_axes(ax)
    names = list(accuracies)
    values = list(accuracies.values())
    palette = [COLORS["neutral"], COLORS["accent_light"], COLORS["accent"], COLORS["positive"]]
    ax.bar([n.replace(" (", "\n(") for n in names], values, color=palette[: len(names)])
    ax.set_ylim(0, 1.08)
    ax.set_title("Accuracy comparison")
    ax.grid(axis="x", visible=False)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
    return fig


def plot_top_words(words, label, ax=None):
    fig, ax = new_axes(ax, figsize=(5, 4))
    ws = [w for w, _ in words][::-1]
    vs = [v for _, v in words][::-1]
    ax.barh(ws, vs, color=color_for_label(label))
    ax.set_title(f"Words most predictive of '{label}'")
    ax.set_xlabel("Model weight (strength)")
    ax.grid(axis="y", visible=False)
    return fig


def plot_confusion(cm, classes, ax=None):
    fig, ax = new_axes(ax, figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    ax.grid(False)
    return fig


def build_dashboard(sentiments: pd.Series, vader_scores: pd.Series, accuracies: dict, result: dict):
    """The six-panel dashboard from the original project (two-class labels)."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("IMDB Movie Review Sentiment Analysis", fontsize=16)
    plot_label_distribution(sentiments.value_counts(), "Sentiment distribution (actual labels)", ax=axes[0, 0])
    plot_vader_hist(vader_scores, ax=axes[0, 1])
    plot_accuracy_compare(accuracies, ax=axes[0, 2])
    pos, neg = result["classes"][1], result["classes"][0]
    plot_top_words(result["top_words"][pos], pos, ax=axes[1, 0])
    plot_top_words(result["top_words"][neg], neg, ax=axes[1, 1])
    plot_confusion(result["confusion"], result["classes"], ax=axes[1, 2])
    fig.tight_layout()
    return fig


# ----------------------------------------------------- original IMDB script
DATA_FILE = "IMDB Dataset.csv"
SAMPLE_SIZE = 10000  # use 50000 for the full dataset (slower)


def main():
    df = pd.read_csv(DATA_FILE).sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} reviews")
    print(df["sentiment"].value_counts(), "\n")

    df["clean_review"] = clean_series(df["review"])
    df = df.join(score_vader(df["clean_review"]))
    df["vader_label"] = df.pop("vader_binary")  # same two-class labels as the original run
    vader_acc = float((df["sentiment"] == df["vader_label"]).mean())
    print(f"VADER accuracy: {vader_acc:.3f}")

    # stratify=False reproduces the split (and the numbers) in the README
    result = train_classifier(df["clean_review"], df["sentiment"], stratify=False)
    uni_acc, ml_acc = result["baseline_accuracy"], result["accuracy"]
    print(f"Logistic Regression (single words) accuracy: {uni_acc:.3f}")
    print(f"Logistic Regression (words + word pairs) accuracy: {ml_acc:.3f}")
    print(f"Improvement from word pairs: {(ml_acc - uni_acc) * 100:+.1f} percentage points\n")
    print(pd.DataFrame(result["report"]).T.round(2), "\n")

    pos, neg = result["classes"][1], result["classes"][0]
    print("Most predictive POSITIVE words:", [w for w, _ in result["top_words"][pos]])
    print("Most predictive NEGATIVE words/phrases:", [w for w, _ in result["top_words"][neg]])

    accuracies = {
        "VADER": vader_acc,
        "LogReg (single words)": uni_acc,
        "LogReg (+ word pairs)": ml_acc,
    }
    fig = build_dashboard(df["sentiment"], df["vader_score"], accuracies, result)
    fig.savefig("sentiment_dashboard.png", dpi=150)
    print("\nSaved chart to sentiment_dashboard.png")

    print("Most FREQUENT positive words:", frequent_words(df[df["sentiment"] == "positive"]["clean_review"]))
    print("Most FREQUENT negative words:", frequent_words(df[df["sentiment"] == "negative"]["clean_review"]))

    wrong = df[df["sentiment"] != df["vader_label"]]
    print(f"\nVADER misclassified {len(wrong)} reviews ({len(wrong) / len(df):.1%}).")
    if len(wrong):
        print("Example misclassified review:\n", wrong.iloc[0]["clean_review"][:400])

    df.to_csv("results.csv", index=False)
    print("Saved full results to results.csv")


if __name__ == "__main__":
    main()
