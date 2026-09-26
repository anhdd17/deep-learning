import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_predict_positive():
    res = client.post("/predict", json={"text": "This movie was absolutely amazing, I loved every second of it!"})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "positive"
    assert 0 < body["probability"] < 1

def test_predict_negative():
    res = client.post("/predict", json={"text": "Terrible movie. Complete waste of time and money."})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == "negative"
    assert 0 < body["probability"] < 1

def test_predict_empty_text():
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 422

def test_predict_whitespace_only():
    res = client.post("/predict", json={"text": "   "})
    assert res.status_code == 422

def test_predict_missing_field():
    res = client.post("/predict", json={})
    assert res.status_code == 422

def test_predict_response_schema():
    res = client.post("/predict", json={"text": "great film"})
    body = res.json()
    assert "label" in body
    assert "probability" in body
    assert body["label"] in ("positive", "negative")
