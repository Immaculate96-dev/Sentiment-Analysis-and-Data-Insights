import pytest

import sentiment_analysis as sa
import tabular_analysis as ta
from demo_data import make_demo_dataframe

pytest.importorskip("vaderSentiment")


@pytest.fixture(scope="module")
def data():
    df, _ = ta.prepare_dataframe(make_demo_dataframe())
    return df, sa.clean_series(df["review_text"])


def test_clean_text():
    assert sa.clean_text("<br />Great!  film  ") == "Great! film"
    assert sa.clean_text(None) == ""


def test_vader_scores_and_labels(data):
    _, texts = data
    out = sa.score_vader(texts)
    assert out["vader_score"].between(-1, 1).all()
    assert set(out["vader_label"]) <= {"positive", "neutral", "negative"}
    assert sa.score_vader(texts.iloc[:1].map(lambda _: "Absolutely wonderful, loved it!"))["vader_label"].iloc[0] == "positive"


def test_classifier_beats_majority_baseline(data):
    df, texts = data
    result = sa.train_classifier(texts, df["sentiment"])
    assert result["accuracy"] > result["majority_accuracy"]
    assert result["n_train"] + result["n_test"] == len(df)
    assert sa.predict_text(result, "Terrible, a waste of money")["label"] == "negative"


def test_multiclass_labels(data):
    df, texts = data
    result = sa.train_classifier(texts, df["rating"].astype(int).astype(str))
    assert len(result["classes"]) == 5 and set(result["top_words"]) == set(result["classes"])


def test_unusable_labels_give_clear_errors(data):
    df, texts = data
    with pytest.raises(ValueError, match="at least two"):
        sa.train_classifier(texts, df["sentiment"].map(lambda _: "same"))
    with pytest.raises(ValueError, match="different values"):
        sa.train_classifier(texts, df["review_id"].astype(str))
    with pytest.raises(ValueError, match="too few"):
        sa.train_classifier(texts.head(10), df["sentiment"].head(10))


def test_label_helpers():
    assert sa.guess_positive_value(["negative", "positive"]) == "positive"
    assert sa.guess_positive_value(["0", "1"]) == "1"
