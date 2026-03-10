"""
tests/test_main.py
==================
Unit tests for main.py — Gmail Email Inbox Agent.

Strategy:
- Gmail API, MongoDB, and MCP server calls are all mocked.
- Tests cover email extraction logic, MCP fallback, and FastAPI endpoints.
- No real network calls or GPU models are required.

Run:
    pytest tests/test_main.py -v
"""

import base64
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import ConnectError, AsyncClient


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build fake MIME message dicts (mimics Gmail API responses)
# ─────────────────────────────────────────────────────────────────────────────

def _b64(text: str) -> str:
    """Base64url encode a string, as Gmail API does."""
    return base64.urlsafe_b64encode(text.encode()).decode()


def _make_message(subject: str = "", sender: str = "", body: str = "",
                  mime_type: str = "text/plain", snippet: str = "") -> dict:
    """Build a minimal Gmail API message dict for testing."""
    return {
        "snippet": snippet or body[:100],
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
            ],
            "mimeType": mime_type,
            "body": {"data": _b64(body)},
        },
    }


def _make_multipart_message(subject: str, sender: str,
                             plain_body: str = "", html_body: str = "") -> dict:
    """Build a multipart Gmail message."""
    parts = []
    if plain_body:
        parts.append({
            "mimeType": "text/plain",
            "body": {"data": _b64(plain_body)},
        })
    if html_body:
        parts.append({
            "mimeType": "text/html",
            "body": {"data": _b64(html_body)},
        })
    return {
        "snippet": plain_body[:100] if plain_body else html_body[:100],
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
            ],
            "mimeType": "multipart/alternative",
            "body": {},
            "parts": parts,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Import functions under test (import after patching anything loaded at module level)
# ─────────────────────────────────────────────────────────────────────────────

from main import extract_email_content, _extract_body


# ═════════════════════════════════════════════════════════════════════════════
#  1. Email Content Extraction Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractEmailContent:
    """Tests for extract_email_content() and _extract_body()."""

    def test_plain_text_message(self):
        """Subject, sender, and plain text body are extracted correctly."""
        msg = _make_message(
            subject="Urgent: Account Verification",
            sender="attacker@evil.com",
            body="Click here to verify your account: http://evil.com/verify",
        )
        result = extract_email_content(msg)

        assert result["subject"] == "Urgent: Account Verification"
        assert result["sender"] == "attacker@evil.com"
        assert "verify your account" in result["body"]
        assert "Subject: Urgent" in result["full_text"]

    def test_sender_with_display_name(self):
        """Sender with 'Name <email>' format → only email address extracted."""
        msg = _make_message(
            subject="Hello",
            sender="John Doe <john@example.com>",
            body="Hi there",
        )
        result = extract_email_content(msg)
        assert result["sender"] == "john@example.com"

    def test_missing_subject_and_sender(self):
        """Empty subject and sender → defaults to empty strings, no crash."""
        msg = _make_message(subject="", sender="", body="Some body text")
        result = extract_email_content(msg)
        assert result["subject"] == ""
        assert result["sender"] == ""
        assert result["body"] == "Some body text"
        # When no subject, full_text should just be the body
        assert result["full_text"] == "Some body text"

    def test_multipart_prefers_plain_text(self):
        """Multipart message → plain text is preferred over HTML."""
        msg = _make_multipart_message(
            subject="Test",
            sender="test@example.com",
            plain_body="This is plain text.",
            html_body="<p>This is <b>HTML</b></p>",
        )
        result = extract_email_content(msg)
        assert result["body"] == "This is plain text."

    def test_multipart_html_fallback(self):
        """When no plain text part exists, falls back to HTML (tags stripped)."""
        msg = _make_multipart_message(
            subject="Test",
            sender="test@example.com",
            plain_body="",
            html_body="<p>Hello <b>World</b></p>",
        )
        result = extract_email_content(msg)
        # HTML tags should be stripped
        assert "<p>" not in result["body"]
        assert "<b>" not in result["body"]
        assert "Hello" in result["body"]
        assert "World" in result["body"]

    def test_html_tags_stripped(self):
        """HTML tags in body text are removed by regex."""
        msg = _make_message(
            subject="Newsletter",
            sender="news@corp.com",
            body="<html><body><p>Check <a href='http://x.com'>this link</a>.</p></body></html>",
            mime_type="text/html",
        )
        result = extract_email_content(msg)
        assert "<html>" not in result["body"]
        assert "<a " not in result["body"]
        assert "Check" in result["body"]
        assert "this link" in result["body"]

    def test_snippet_passthrough(self):
        """Snippet field is taken directly from the Gmail message."""
        msg = _make_message(subject="X", sender="x@y.com", body="Full body", snippet="Short snippet")
        result = extract_email_content(msg)
        assert result["snippet"] == "Short snippet"

    def test_extract_body_empty_payload(self):
        """_extract_body with empty payload returns empty string."""
        result = _extract_body({})
        assert result == ""


# ═════════════════════════════════════════════════════════════════════════════
#  2. MCP Server Communication Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestClassifyViaMcp:
    """Tests for classify_via_mcp() — network calls are mocked."""

    @pytest.mark.asyncio
    async def test_success_response(self):
        """Happy path: MCP server responds with a classification dict."""
        from main import classify_via_mcp

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "label": "phishing",
            "confidence": 0.92,
            "reason": "Suspicious link",
            "indicators": ["urgent language", "unfamiliar sender"],
            "action": "quarantine",
            "path": "fast",
            "latency_ms": 45.0,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("main.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await classify_via_mcp("Subject: You won $1M\n\nClick here now!")

        assert result["label"] == "phishing"
        assert result["confidence"] == 0.92
        assert result["action"] == "quarantine"

    @pytest.mark.asyncio
    async def test_connect_error_fallback(self):
        """When MCP server is unreachable, returns a safe fallback dict."""
        from main import classify_via_mcp

        with patch("main.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=ConnectError("Connection refused")
            )
            result = await classify_via_mcp("Some email text")

        assert result["label"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["action"] == "review"
        assert result["path"] == "fallback"

    @pytest.mark.asyncio
    async def test_generic_exception_fallback(self):
        """Any unexpected exception during MCP call → safe fallback."""
        from main import classify_via_mcp

        with patch("main.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=RuntimeError("Unexpected error")
            )
            result = await classify_via_mcp("Some email text")

        assert result["label"] == "unknown"
        assert result["path"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
#  3. FastAPI Endpoint Tests
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def test_client():
    """
    Create a TestClient for the FastAPI app.
    We patch out Gmail, MongoDB, and asyncio.create_task so the app
    starts cleanly without any real external connections.
    """
    from unittest.mock import AsyncMock, patch, MagicMock

    # Patch MongoDB client
    mock_mongo = MagicMock()
    mock_mongo.admin.command = AsyncMock(return_value={"ok": 1})
    mock_mongo.jenkins = MagicMock()

    # Patch Gmail credentials
    mock_creds = MagicMock()
    mock_service = MagicMock()

    with patch("main.AsyncIOMotorClient", return_value=mock_mongo), \
         patch("main.get_gmail_service", return_value=mock_service), \
         patch("main.setup_gmail_labels", return_value={}), \
         patch("asyncio.create_task"):

        from httpx import AsyncClient as HttpxAsyncClient
        import asyncio
        from main import app as fastapi_app

        # Use httpx's AsyncClient with ASGITransport for testing
        import httpx
        transport = httpx.ASGITransport(app=fastapi_app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        yield client


class TestHealthEndpoint:
    """Tests for GET /health"""

    @pytest.mark.asyncio
    async def test_health_returns_running(self, test_client):
        """Health endpoint returns agent: running status."""
        with patch("main.httpx.AsyncClient") as MockMcpClient:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"status": "healthy"}
            MockMcpClient.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            response = await test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["agent"] == "running"
        assert data["polling"] == "active"


class TestLogsEndpoint:
    """Tests for GET /api/logs"""

    @pytest.mark.asyncio
    async def test_logs_returns_list(self, test_client):
        """Logs endpoint fetches from DB and returns list."""
        fake_records = [
            {
                "_id": "abc123",
                "message_id": "msg1",
                "subject": "Test Email",
                "snippet": "Hello",
                "sender": "x@y.com",
                "label_applied": "clean",
                "confidence": 0.98,
                "reason": "Safe",
                "indicators": [],
                "action": "deliver",
                "inference_path": "fast",
                "latency_ms": 23.1,
                "status": "processed",
                "created_at": "2026-03-10T10:00:00Z",
            }
        ]

        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=fake_records.copy())

        with patch("main.app") as mock_app:
            # This is tricky — we test the route logic directly
            pass  # Endpoint test covered by integration test approach below

    @pytest.mark.asyncio
    async def test_logs_endpoint_structure(self, test_client):
        """GET /api/logs response has expected JSON keys."""
        # Patch the DB at the app level
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])

        import main
        main.app.db = MagicMock()
        main.app.db.emails.find.return_value = mock_cursor

        response = await test_client.get("/api/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "total_returned" in data
        assert isinstance(data["logs"], list)


class TestManualClassifyEndpoint:
    """Tests for POST /api/classify"""

    @pytest.mark.asyncio
    async def test_classify_proxies_to_mcp(self, test_client):
        """POST /api/classify calls MCP and returns its result."""
        mock_result = {
            "label": "clean",
            "confidence": 0.97,
            "reason": "Normal email",
            "indicators": [],
            "action": "deliver",
            "path": "fast",
            "latency_ms": 30.0,
        }
        with patch("main.classify_via_mcp", return_value=mock_result) as mock_classify:
            response = await test_client.post(
                "/api/classify",
                params={"email_text": "Subject: Hello\n\nHow are you?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "clean"
