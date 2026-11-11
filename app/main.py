"""NewsLens API — classify news text into a topic with TF-IDF + LogisticRegression.

Endpoints:
    GET  /          — small HTML UI: paste text, see the predicted topic.
    GET  /health    — {"status": "ok", "model_loaded": true/false}
    GET  /topics    — list of the 4 topics the model predicts.
    POST /classify  — {"text": "..."} ->
                     {"topic": ..., "confidence": ..., "top_predictions": [...]}

Run:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

# Filled in at startup by load_model(); kept as module globals so the
# /classify handler stays simple.
_artifacts = {"vectorizer": None, "model": None, "topics": []}


def load_model():
    """Load the trained vectorizer + classifier saved by train.py."""
    bundle = joblib.load(MODEL_PATH)
    _artifacts["vectorizer"] = bundle["vectorizer"]
    _artifacts["model"] = bundle["model"]
    _artifacts["topics"] = bundle["topics"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="NewsLens",
    description="News topic classifier — TF-IDF + LogisticRegression (scikit-learn).",
    lifespan=lifespan,
)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="News text to classify.")


class TopicScore(BaseModel):
    topic: str
    confidence: float


class ClassifyResponse(BaseModel):
    topic: str
    confidence: float
    top_predictions: list[TopicScore]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _artifacts["model"] is not None}


@app.get("/topics")
def topics():
    return {"topics": _artifacts["topics"]}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    vectorizer = _artifacts["vectorizer"]
    model = _artifacts["model"]

    # predict_proba columns follow model.classes_ (alphabetical for string
    # labels), NOT the saved TOPICS order — pair them via classes_ so each
    # probability gets its true topic.
    classes = list(model.classes_)
    probs = model.predict_proba(vectorizer.transform([req.text]))[0]
    ranked = sorted(zip(classes, probs), key=lambda pair: pair[1], reverse=True)
    best_topic, best_conf = ranked[0]
    return ClassifyResponse(
        topic=best_topic,
        confidence=float(best_conf),
        top_predictions=[
            TopicScore(topic=topic, confidence=float(prob)) for topic, prob in ranked
        ],
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NewsLens — News Topic Classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 3rem auto; padding: 0 1rem; }
    h1 { font-size: 1.6rem; }
    textarea { width: 100%; min-height: 140px; font: inherit; padding: 0.6rem; }
    button { margin-top: 0.6rem; padding: 0.6rem 1.2rem; font: inherit; cursor: pointer; }
    #result { margin-top: 1rem; padding: 1rem; border: 1px solid #ccc; border-radius: 8px; display: none; }
    #result h2 { margin: 0 0 0.4rem; text-transform: capitalize; }
    #result li { text-transform: capitalize; }
  </style>
</head>
<body>
  <h1>📰 NewsLens</h1>
  <p>Paste a piece of news text and the model will guess its topic:
  sports, technology, politics, or science.</p>
  <textarea id="text" placeholder="Paste news text here..."></textarea>
  <br>
  <button onclick="classify()">Classify</button>
  <div id="result"></div>
  <script>
    async function classify() {
      const text = document.getElementById("text").value.trim();
      const result = document.getElementById("result");
      if (!text) { result.style.display = "block"; result.innerHTML = "Please paste some text first."; return; }
      const res = await fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      result.style.display = "block";
      result.innerHTML = "<h2>" + data.topic + "</h2>"
        + "<p>Confidence: " + (data.confidence * 100).toFixed(1) + "%</p>"
        + "<ul>" + data.top_predictions.map(p =>
            "<li>" + p.topic + ": " + (p.confidence * 100).toFixed(1) + "%</li>").join("") + "</ul>";
    }
  </script>
</body>
</html>
"""
