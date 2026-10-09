"""
Sentiment Analysis Tool - IMDB Movie Reviews
--------------------------------------------
Setup (run once in your terminal):
    pip install pandas matplotlib scikit-learn vaderSentiment

Data:
    Download "IMDB Dataset.csv" from Kaggle (IMDB Dataset of 50K Movie Reviews)
    and put it in the same folder as this script.
    Columns: review, sentiment (positive/negative)

Run:
    python sentiment_analysis.py
"""

import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_FILE = "IMDB Dataset.csv"
SAMPLE_SIZE = 10000  # use 50000 for the full dataset (slower)

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv(DATA_FILE).sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"Loaded {len(df)} reviews")
print(df["sentiment"].value_counts(), "\n")


# ---------------------------------------------------------------- 2. Clean
def clean(text):
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags like <br />
    text = re.sub(r"\s+", " ", text)    # collapse extra whitespace
    return text.strip()


df["clean_review"] = df["review"].apply(clean)

# ---------------------------------------------------------------- 3. VADER
analyzer = SentimentIntensityAnalyzer()
df["vader_score"] = df["clean_review"].apply(
    lambda t: analyzer.polarity_scores(t)["compound"]
)
# compound score ranges from -1 (very negative) to +1 (very positive)
df["vader_label"] = df["vader_score"].apply(
    lambda s: "positive" if s >= 0 else "negative"
)

vader_acc = accuracy_score(df["sentiment"], df["vader_label"])
print(f"VADER accuracy: {vader_acc:.3f}")

# ---------------------------------------------------------------- 4. ML model
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["sentiment"], test_size=0.2, random_state=42
)
# --- Model A: single words only (baseline)
vec_uni = TfidfVectorizer(stop_words="english", max_features=20000)
model_uni = LogisticRegression(max_iter=1000)
model_uni.fit(vec_uni.fit_transform(X_train), y_train)
uni_acc = accuracy_score(y_test, model_uni.predict(vec_uni.transform(X_test)))
print(f"Logistic Regression (single words) accuracy: {uni_acc:.3f}")

# --- Model B: single words + two-word pairs (can see things like "not good")
# Standard stop-word lists remove "not", "no", "never", which would destroy the
# very thing we want to capture, so we keep the negation words.
NEGATIONS = {"not", "no", "nor", "never", "cannot", "nothing", "nobody", "neither"}
custom_stops = list(ENGLISH_STOP_WORDS - NEGATIONS)

vectorizer = TfidfVectorizer(
    stop_words=custom_stops, ngram_range=(1, 2), min_df=3, max_features=50000
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
ml_pred = model.predict(X_test_vec)
ml_acc = accuracy_score(y_test, ml_pred)
print(f"Logistic Regression (words + word pairs) accuracy: {ml_acc:.3f}")
print(f"Improvement from word pairs: {(ml_acc - uni_acc) * 100:+.1f} percentage points\n")
print(classification_report(y_test, ml_pred))

# ---------------------------------------------------------------- 5. Predictive words
# The model gives every word a weight. Big positive weight = pushes a review
# towards "positive"; big negative weight = pushes it towards "negative".
feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]  # classes_ is ['negative', 'positive'], so + = positive
order = weights.argsort()
top_neg = [(feature_names[i], weights[i]) for i in order[:12]]
top_pos = [(feature_names[i], weights[i]) for i in order[::-1][:12]]
print("\nMost predictive POSITIVE words:", [w for w, _ in top_pos])
print("Most predictive NEGATIVE words/phrases:", [w for w, _ in top_neg])

# ---------------------------------------------------------------- 6. Charts
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("IMDB Movie Review Sentiment Analysis", fontsize=16)

# a) label distribution
df["sentiment"].value_counts().plot(
    kind="bar", ax=axes[0, 0], color=["#4c9f70", "#d9534f"]
)
axes[0, 0].set_title("Sentiment distribution (actual labels)")
axes[0, 0].set_xlabel("")
axes[0, 0].tick_params(axis="x", rotation=0)

# b) VADER score histogram
axes[0, 1].hist(df["vader_score"], bins=40, color="#5b7db1")
axes[0, 1].set_title("Distribution of VADER scores")
axes[0, 1].set_xlabel("Compound score (-1 to +1)")

# c) model comparison
accs = [vader_acc, uni_acc, ml_acc]
axes[0, 2].bar(["VADER", "LogReg\n(single words)", "LogReg\n(+ word pairs)"], accs,
               color=["#e0a030", "#8fa8d0", "#5b7db1"])
axes[0, 2].set_ylim(0, 1)
axes[0, 2].set_title("Model accuracy comparison")
for i, v in enumerate(accs):
    axes[0, 2].text(i, v + 0.02, f"{v:.1%}", ha="center")

# d) most predictive positive words
axes[1, 0].barh([w for w, _ in top_pos][::-1], [c for _, c in top_pos][::-1],
                color="#4c9f70")
axes[1, 0].set_title("Words/phrases most predictive of POSITIVE")
axes[1, 0].set_xlabel("Model weight")

# e) most predictive negative words
axes[1, 1].barh([w for w, _ in top_neg][::-1], [abs(c) for _, c in top_neg][::-1],
                color="#d9534f")
axes[1, 1].set_title("Words/phrases most predictive of NEGATIVE")
axes[1, 1].set_xlabel("Model weight (strength)")

# f) confusion matrix for the ML model
ConfusionMatrixDisplay.from_predictions(
    y_test, ml_pred, ax=axes[1, 2], cmap="Blues", colorbar=False
)
axes[1, 2].set_title("Logistic Regression confusion matrix")

plt.tight_layout()
plt.savefig("sentiment_dashboard.png", dpi=150)
print("\nSaved chart to sentiment_dashboard.png")

# Most common words overall (for comparison with the predictive words above)
def top_words(texts, n=10):
    words = re.findall(r"[a-z']+", " ".join(texts).lower())
    extra = {"movie", "film", "br", "one", "like", "just", "really", "it's"}
    words = [w for w in words
             if w not in ENGLISH_STOP_WORDS and w not in extra and len(w) > 2]
    return Counter(words).most_common(n)


print("Most FREQUENT positive words:", top_words(df[df["sentiment"] == "positive"]["clean_review"]))
print("Most FREQUENT negative words:", top_words(df[df["sentiment"] == "negative"]["clean_review"]))

# ---------------------------------------------------------------- 7. Insights
# Reviews VADER got wrong are interesting for your report (sarcasm, mixed reviews)
wrong = df[df["sentiment"] != df["vader_label"]]
print(f"\nVADER misclassified {len(wrong)} reviews ({len(wrong)/len(df):.1%}).")
print("Example misclassified review:\n", wrong.iloc[0]["clean_review"][:400])

df.to_csv("results.csv", index=False)
print("Saved full results to results.csv")
