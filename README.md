# NewsLens 📰

A news topic classifier API. Paste in a piece of news text and it predicts the
topic — **sports, technology, politics, or science** — with a confidence score.

Built with **FastAPI** and **scikit-learn** (TF-IDF + LogisticRegression),
trained on the classic 20 Newsgroups dataset.

## How it works

- **Data:** the [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)
  dataset, restricted to 4 categories — one per topic:
  - `rec.sport.baseball` → sports
  - `comp.sys.mac.hardware` → technology
  - `talk.politics.misc` → politics
  - `sci.space` → science
- **Model:** TF-IDF vectorizer (word unigrams + bigrams, `max_features=3000`,
  English stop words removed) feeding a LogisticRegression classifier.
- **Accuracy: 83.2%** on the 1,486-document held-out test split
  (2,233 training docs). Per-topic F1: sports 0.84, technology 0.89,
  politics 0.80, science 0.80.
- The trained vectorizer + model are saved to `model/model.pkl` (204 KB) and
  ship in the repo, so the API runs out of the box — no retraining needed.

## Run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the built-in UI, or use the API:

```bash
# Health check
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true}

# List topics
curl http://127.0.0.1:8000/topics
# {"topics":["sports","technology","politics","science"]}

# Classify some text
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launched a new satellite into orbit to study the surface of Mars."}'
# {"topic":"science","confidence":0.99,"top_predictions":[
#   {"topic":"science","confidence":0.99},
#   {"topic":"politics","confidence":0.0},
#   {"topic":"sports","confidence":0.0},
#   {"topic":"technology","confidence":0.0}]}
```

## Retrain the model

```bash
python train.py
```

This re-downloads 20 Newsgroups, retrains, prints the accuracy and
classification report, and overwrites `model/model.pkl`.

## Tests

```bash
pytest
```

7 tests: `/health`, `/topics`, `/classify` on sports and technology text
(sensible topic, confidence in 0–1, ranked `top_predictions` summing to 1),
invalid input returning 422 (missing and empty text), and the HTML UI at `/`.

## Project structure

```
newslens/
├── train.py            # fetch 20 Newsgroups, train TF-IDF + LogisticRegression, save model
├── app/
│   ├── __init__.py
│   └── main.py         # FastAPI app: /classify, /health, /topics, HTML UI at /
├── model/
│   └── model.pkl       # trained vectorizer + classifier (204 KB)
├── tests/
│   └── test_api.py     # pytest suite (7 tests)
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
