"""
Integration tests for logitguard.app.server endpoints
Tests /samples, /rescore, /benchmarks, and Contract C round-tripping.
"""

import pytest
from fastapi.testclient import TestClient
from app.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_server_samples_endpoint(client):
    res = client.get("/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) >= 50
    # Every sample must have id and response_preview
    for s in data["samples"]:
        assert "id" in s
        assert "task" in s
        assert "response_preview" in s


def test_server_rescore_endpoint(client):
    res = client.post(
        "/rescore",
        json={
            "sample_id": "sample_d2t_00",
            "tau": 0.68,
            "variant": "core",
            "mask_mode": "soft",
        },
    )
    assert res.status_code == 200
    data = res.json()
    # Contract C required keys
    assert data["id"] == "sample_d2t_00"
    assert "spans" in data
    assert "tokens" in data
    assert "metrics" in data
    assert "timing_ms" in data
    assert data["tau"] == 0.68
    assert data["variant"] == "core"
    assert data["mask_mode"] == "soft"


def test_server_benchmarks_endpoint(client):
    res = client.get("/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert "main_table" in data
    assert "ablation_table" in data
    assert "latency_table" in data
    assert len(data["main_table"]) > 0
    assert len(data["ablation_table"]) > 0
    assert len(data["latency_table"]) > 0


def test_server_health_and_status(client):
    res_h = client.get("/health")
    assert res_h.status_code == 200
    data_h = res_h.json()
    assert "status" in data_h
    assert "ready" in data_h

    res_s = client.get("/status")
    assert res_s.status_code == 200
    data_s = res_s.json()
    assert "status" in data_s
    assert "model" in data_s


def test_server_api_run_validation(client):
    # Empty query should return 422 Unprocessable Entity
    res = client.post("/api/run", json={"query": "   ", "context": ""})
    assert res.status_code == 422
