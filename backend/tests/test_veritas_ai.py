import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.ml import predictor
from app.api import routes

client = TestClient(app)


def test_health_endpoint_veritas_ai_branding():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Veritas AI" in data["platform"]
    assert "available_models" in data
    assert isinstance(data["models"], list)


def test_predict_endpoint_empty_headline_rejected():
    response = client.post(
        "/api/predict",
        json={"headline": "   ", "article": " ".join(["word"] * 60), "model": "dense"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "EMPTY_HEADLINE"


def test_predict_endpoint_short_article_rejected():
    response = client.post(
        "/api/predict",
        json={"headline": "Valid Headline", "article": "Too short article text", "model": "dense"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "ARTICLE_TOO_SHORT"


def test_predict_news_enriched_response(monkeypatch):
    monkeypatch.setattr(predictor, "model_inventory", lambda: [{"name": "dense", "available": True}])
    monkeypatch.setattr(predictor, "predict_texts", lambda _model, _texts: np.array([[0.1, 0.9]]))
    monkeypatch.setattr(
        predictor,
        "read_json",
        lambda _path: {"class_to_index": {"FAKE": 0, "REAL": 1}},
    )

    result = predictor.predict_news("Global Market Rally", " ".join(["article"] * 60), "dense")
    assert result["prediction"] == "REAL"
    assert result["predicted_label_display"] == "Predicted Real"
    assert result["confidence"] == 0.9
    assert result["real_probability"] == 0.9
    assert result["fake_probability"] == 0.1
    assert result["article_words"] == 60
    assert "cleaned_text_preview" in result
    assert "timestamp" in result


def test_explain_endpoint_response_structure_and_caching(monkeypatch):
    routes._EXPLANATION_CACHE.clear()

    monkeypatch.setattr(routes, "MODEL_NAMES", ("dense", "rnn", "lstm", "gru", "bert"))
    monkeypatch.setattr(
        routes,
        "read_json",
        lambda _path: {
            "class_to_index": {"FAKE": 0, "REAL": 1},
            "index_to_class": {"0": "FAKE", "1": "REAL"},
            "sigmoid_positive_label": "REAL",
            "sequence_length": 500,
        },
    )

    # Mock predict_texts to return realistic probabilities: [FAKE prob, REAL prob]
    def mock_predict(model, texts):
        res = []
        for text in texts:
            # If text has "verified" or "reuters", give high real prob
            real_score = 0.85 if "verified" in text.lower() else 0.70
            res.append([1.0 - real_score, real_score])
        return np.array(res)

    monkeypatch.setattr(routes, "predict_texts", mock_predict)

    payload = {
        "headline": "Verified economic indicators show quarterly expansion",
        "article": "The national statistics department confirmed on Friday that gross domestic product rose solidly after months of sustained industrial growth across multiple commercial sectors. Leading economists agreed that verified trade indicators reflect ongoing fiscal resilience and sustainable enterprise stability throughout major export markets, confirming that recent municipal investments in infrastructure have delivered tangible productivity dividends across regional hubs.",
        "model": "dense",
    }

    # First call (uncached)
    response = client.post("/api/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "dense"
    assert data["prediction"] in {"REAL", "FAKE"}
    assert "confidence" in data
    assert isinstance(data["words"], list)
    assert "plain_english_summary" in data
    assert "methodology" in data
    assert data["cached"] is False
    assert data["execution_time_ms"] >= 0

    # Second call (must be served from cache)
    second_response = client.post("/api/explain", json=payload)
    assert second_response.status_code == 200
    second_data = second_response.json()
    assert second_data["cached"] is True
