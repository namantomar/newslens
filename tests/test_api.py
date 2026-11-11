"""API tests for NewsLens.

Uses the model trained by train.py (model/model.pkl). Run train.py first,
then `pytest` from the project root.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="model/model.pkl missing — run `python train.py` first",
)

from app.main import app  # noqa: E402

VALID_TOPICS = {"sports", "technology", "politics", "science"}


@pytest.fixture(scope="module")
def client():
    # Entering the context manager triggers the app's lifespan handler,
    # which loads the trained model. A plain TestClient(app) does not.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["model_loaded"] is True


def test_topics(client):
    res = client.get("/topics")
    assert res.status_code == 200
    assert set(res.json()["topics"]) == VALID_TOPICS


def test_classify_sports(client):
    res = client.post(
        "/classify",
        json={"text": "The pitcher struck out ten batters as the home team won the baseball game in extra innings."},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["topic"] in VALID_TOPICS
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["topic"] == "sports"
    assert len(data["top_predictions"]) == 4
    # top_predictions is sorted best-first and sums to ~1
    confidences = [p["confidence"] for p in data["top_predictions"]]
    assert confidences == sorted(confidences, reverse=True)
    assert abs(sum(confidences) - 1.0) < 1e-6


def test_classify_technology(client):
    res = client.post(
        "/classify",
        json={"text": "The new laptop ships with a faster processor, more memory, and a redesigned cooling system for the motherboard."},
    )
    assert res.status_code == 200
    assert res.json()["topic"] in VALID_TOPICS


def test_classify_missing_text_returns_422(client):
    res = client.post("/classify", json={})
    assert res.status_code == 422


def test_classify_empty_text_returns_422(client):
    res = client.post("/classify", json={"text": ""})
    assert res.status_code == 422


def test_index_returns_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "NewsLens" in res.text
