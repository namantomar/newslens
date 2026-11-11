"""Train the NewsLens topic classifier.

Fetches the 20 Newsgroups dataset (4 categories, one per topic), trains a
TF-IDF vectorizer + LogisticRegression classifier, evaluates it on the held-out
test split, and saves the fitted vectorizer + model to model/model.pkl.

Usage:
    python train.py
"""

import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# One 20 Newsgroups category per topic. These four are clearly separable,
# which keeps the demo classifier accurate on a small feature budget.
CATEGORY_TO_TOPIC = {
    "rec.sport.baseball": "sports",
    "comp.sys.mac.hardware": "technology",
    "talk.politics.misc": "politics",
    "sci.space": "science",
}

TOPICS = ["sports", "technology", "politics", "science"]
MODEL_PATH = "model/model.pkl"


def load_data(subset):
    """Fetch one split of 20 Newsgroups and map its labels to topic names."""
    categories = list(CATEGORY_TO_TOPIC.keys())
    data = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=("headers", "footers", "quotes"),  # strip metadata, keep the text
    )
    # NOTE: data.target indexes into data.target_names, which sklearn returns
    # in sorted order — NOT the order of `categories` above. Map through
    # target_names so each document gets its true topic.
    topics = [CATEGORY_TO_TOPIC[data.target_names[label]] for label in data.target]
    return data.data, topics


def main():
    print("Downloading 20 Newsgroups (train split)...")
    X_train, y_train = load_data("train")
    print("Downloading 20 Newsgroups (test split)...")
    X_test, y_test = load_data("test")
    print(f"Train docs: {len(X_train)}  Test docs: {len(X_test)}")

    # TF-IDF: word unigrams + bigrams, capped at 3000 features so the saved
    # model stays small.
    vectorizer = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=10)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    # labels=TOPICS pins the row order (string labels would otherwise sort).
    print(classification_report(y_test, y_pred, labels=TOPICS, target_names=TOPICS))

    joblib.dump({"vectorizer": vectorizer, "model": model, "topics": TOPICS}, MODEL_PATH)
    print(f"Saved model + vectorizer to {MODEL_PATH}")


if __name__ == "__main__":
    main()
