"""
tests/test_model_server.py
==========================
Unit tests for model_server.py — MCP FastAPI service that hosts all 3 models.

Strategy:
- The global `classifier` object is mocked — no model weights or GPU needed.
- Tests cover: health checks, 503 guard, single classify, force_deep, batch, and /models list.

Run:
    pytest tests/test_model_server.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_classifier(label: str = "phishing", confidence: float = 0.93,
                           path: str = "fast", force_deep_label: str = "phishing"):
    """Build a mock EmailClassifier that returns deterministic results."""
    clf = MagicMock()
    clf.classifier_device = "cuda:0"
    clf.gemma_device = "cuda:0"
    clf.temperature = 1.5
    clf.threshold = 0.75
    clf.id2label = {0: "clean", 1: "spam", 2: "phishing",
                    3: "fraud", 4: "malware", 5: "threat"}

    clf.classify.return_value = {
        "label": label,
        "confidence": confidence,
        "reason": "High-confidence classification",
        "indicators": [],
        "action": "quarantine",
        "path": path,
    }
    clf._classify_deep.return_value = {
        "label": force_deep_label,
        "confidence": 0.85,
        "reason": "Deep analysis result",
        "indicators": ["suspicious_url"],
        "action": "quarantine",
        "path": "deep (forced)",
    }
    return clf


@pytest.fixture
def server_client():
    """
    TestClient for model_server.py.
    We patch the lifespan so models are never actually loaded,
    and inject a mock classifier directly.
    """
    import httpx
    import model_server

    mock_clf = _make_mock_classifier()

    # Patch the global classifier
    with patch.object(model_server, "classifier", mock_clf):
        # Override lifespan so no real model loading happens
        from contextlib import asynccontextmanager
        from fastapi import FastAPI

        @asynccontextmanager
        async def mock_lifespan(app):
            yield

        # Rebuild app with mock lifespan
        test_app = FastAPI(lifespan=mock_lifespan)
        # Mount routes from the real app
        for route in model_server.app.routes:
            test_app.routes.append(route)

        transport = httpx.ASGITransport(app=model_server.app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        yield client, mock_clf


# ═════════════════════════════════════════════════════════════════════════════
#  1. Health Endpoint Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """GET /health"""

    @pytest.mark.asyncio
    async def test_health_when_classifier_loaded(self, server_client):
        """Returns status=healthy when classifier is not None."""
        client, _ = server_client
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

    @pytest.mark.asyncio
    async def test_health_when_classifier_none(self):
        """Returns status=loading when classifier is None."""
        import model_server
        import httpx

        with patch.object(model_server, "classifier", None):
            transport = httpx.ASGITransport(app=model_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loading"
        assert data["models_loaded"] is False


# ═════════════════════════════════════════════════════════════════════════════
#  2. Classify Endpoint Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifyEndpoint:
    """POST /classify"""

    @pytest.mark.asyncio
    async def test_503_when_models_loading(self):
        """Returns 503 Service Unavailable if classifier not yet loaded."""
        import model_server
        import httpx

        with patch.object(model_server, "classifier", None):
            transport = httpx.ASGITransport(app=model_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/classify",
                    json={"email_text": "Test email", "force_deep": False}
                )

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_classify_success_fast_path(self, server_client):
        """POST /classify returns full ClassifyResponse on success."""
        client, mock_clf = server_client
        response = await client.post(
            "/classify",
            json={
                "email_text": "Subject: URGENT! Your account is suspended\n\nClick here now!",
                "force_deep": False,
            }
        )
        assert response.status_code == 200
        data = response.json()
        # ClassifyResponse fields
        assert "label" in data
        assert "confidence" in data
        assert "reason" in data
        assert "indicators" in data
        assert "action" in data
        assert "path" in data
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], float)
        # Classifier was called
        mock_clf.classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_force_deep_bypasses_gate(self, server_client):
        """force_deep=True calls _classify_deep() directly, not classify()."""
        client, mock_clf = server_client
        response = await client.post(
            "/classify",
            json={
                "email_text": "Ambiguous email text here",
                "force_deep": True,
            }
        )
        assert response.status_code == 200
        # _classify_deep called, NOT classify
        mock_clf._classify_deep.assert_called_once()
        mock_clf.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_response_has_latency(self, server_client):
        """latency_ms field is a positive number."""
        client, _ = server_client
        response = await client.post(
            "/classify",
            json={"email_text": "Hello, how are you?", "force_deep": False}
        )
        data = response.json()
        assert data["latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_classify_label_is_string(self, server_client):
        """label field is always a string."""
        client, _ = server_client
        response = await client.post(
            "/classify",
            json={"email_text": "Click here for free money!", "force_deep": False}
        )
        data = response.json()
        assert isinstance(data["label"], str)
        assert data["label"] in ["clean", "spam", "phishing", "fraud", "malware", "threat", "unknown"]


# ═════════════════════════════════════════════════════════════════════════════
#  3. Batch Classify Endpoint Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBatchClassifyEndpoint:
    """POST /classify/batch"""

    @pytest.mark.asyncio
    async def test_batch_503_when_loading(self):
        """Returns 503 if classifier not loaded."""
        import model_server
        import httpx

        with patch.object(model_server, "classifier", None):
            transport = httpx.ASGITransport(app=model_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/classify/batch",
                    json={"emails": [{"email_text": "test", "force_deep": False}]}
                )

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_batch_classify_returns_all_results(self, server_client):
        """Batch of N emails → N results in response."""
        client, mock_clf = server_client
        emails = [
            {"email_text": "Email one — buy now!", "force_deep": False},
            {"email_text": "Email two — meeting at 3pm", "force_deep": False},
            {"email_text": "Email three — verify account", "force_deep": False},
        ]
        response = await client.post("/classify/batch", json={"emails": emails})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_latency_ms" in data
        assert len(data["results"]) == 3

    @pytest.mark.asyncio
    async def test_batch_result_shape(self, server_client):
        """Each result in batch has all expected ClassifyResponse fields."""
        client, _ = server_client
        response = await client.post(
            "/classify/batch",
            json={"emails": [{"email_text": "Sample email", "force_deep": False}]}
        )
        data = response.json()
        result = data["results"][0]
        for field in ["label", "confidence", "reason", "indicators", "action", "path", "latency_ms"]:
            assert field in result, f"Missing field '{field}' in batch result"

    @pytest.mark.asyncio
    async def test_batch_force_deep_per_email(self, server_client):
        """force_deep=True on individual email in batch calls _classify_deep."""
        client, mock_clf = server_client
        mock_clf.classify.reset_mock()
        mock_clf._classify_deep.reset_mock()

        response = await client.post(
            "/classify/batch",
            json={"emails": [{"email_text": "Suspicious email", "force_deep": True}]}
        )
        assert response.status_code == 200
        mock_clf._classify_deep.assert_called_once()
        mock_clf.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_batch(self, server_client):
        """Empty emails list → results list is empty, total_latency_ms >= 0."""
        client, _ = server_client
        response = await client.post("/classify/batch", json={"emails": []})
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total_latency_ms"] >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  4. Models List Endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestModelsEndpoint:
    """GET /models"""

    @pytest.mark.asyncio
    async def test_models_returns_3_models(self, server_client):
        """Returns exactly 3 model entries: RoBERTa, gate, and Gemma."""
        client, _ = server_client
        response = await client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 3

    @pytest.mark.asyncio
    async def test_models_has_expected_names(self, server_client):
        """Model entries have correct name fields."""
        client, _ = server_client
        response = await client.get("/models")
        data = response.json()
        names = [m["name"] for m in data["models"]]
        assert "roberta-classifier" in names
        assert "open-set-gate" in names
        assert "gemma-2b-lora" in names

    @pytest.mark.asyncio
    async def test_models_loading_when_none(self):
        """Returns {'status': 'loading'} when classifier not ready."""
        import model_server
        import httpx

        with patch.object(model_server, "classifier", None):
            transport = httpx.ASGITransport(app=model_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "loading"
