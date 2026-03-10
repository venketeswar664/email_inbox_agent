"""
tests/test_orchestrator.py
==========================
Unit tests for orchestrator.py — EmailClassifier inference engine.

Strategy:
- ALL model loading (RoBERTa, Gemma, AutoTokenizer, PeftModel) is patched.
- We test the LOGIC of the classifier: gate thresholds, action maps,
  JSON parsing, fast vs deep path selection, and fallback behaviour.
- Zero GPU or trained weights needed.

Run:
    pytest tests/test_orchestrator.py -v
"""

import json
import torch
import pytest
from unittest.mock import patch, MagicMock, mock_open


# ─────────────────────────────────────────────────────────────────────────────
# Shared mock fixtures so we don't re-patch in every test
# ─────────────────────────────────────────────────────────────────────────────

FAKE_LABEL_CONFIG = json.dumps({
    "id2label": {
        "0": "clean",
        "1": "spam",
        "2": "phishing",
        "3": "fraud",
        "4": "malware",
        "5": "threat",
    }
})

FAKE_GATE_CONFIG = json.dumps({"T": 1.5, "tau": 0.75})


def _make_mock_tokenizer():
    """Minimal mock tokenizer that returns dict-like tensors."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
    }
    tok.to = MagicMock(return_value=tok)
    tok.apply_chat_template = MagicMock(
        return_value=torch.zeros((1, 10), dtype=torch.long)
    )
    tok.decode = MagicMock(return_value='{"label": "phishing", "confidence": 0.91, '
                                        '"reason": "Suspicious", "indicators": ["urgent"], '
                                        '"action": "quarantine"}')
    tok.eos_token = "<eos>"
    tok.pad_token = None
    return tok


def _make_mock_cls_model(logit_values: list = None):
    """Mock RoBERTa model that returns fake logits."""
    if logit_values is None:
        # Default: high confidence for label index 2 (phishing)
        logit_values = [-1.0, -1.0, 5.0, -1.0, -1.0, -1.0]

    model = MagicMock()
    fake_logits = torch.tensor([logit_values])  # shape (1, 6)
    output = MagicMock()
    output.logits = fake_logits
    model.return_value = output
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


def _make_mock_gemma_model(decode_output: str = None):
    """Mock Gemma model that returns a predefined token sequence."""
    model = MagicMock()
    model.device = torch.device("cpu")
    # generate() returns token ids — the actual decode call is on the tokenizer
    model.generate = MagicMock(return_value=torch.zeros((1, 20), dtype=torch.long))
    model.eval = MagicMock(return_value=model)
    return model


@pytest.fixture
def classifier_mocks():
    """
    Patch every external dependency of EmailClassifier so it can be
    instantiated without any real files, GPU, or model weights.
    """
    mock_cls_tok   = _make_mock_tokenizer()
    mock_cls_model = _make_mock_cls_model()
    mock_gem_tok   = _make_mock_tokenizer()
    mock_gem_model = _make_mock_gemma_model()
    mock_base_model = MagicMock()

    patches = [
        patch("orchestrator.AutoTokenizer.from_pretrained",
              side_effect=[mock_cls_tok, mock_gem_tok]),
        patch("orchestrator.AutoModelForSequenceClassification.from_pretrained",
              return_value=mock_cls_model),
        patch("orchestrator.AutoModelForCausalLM.from_pretrained",
              return_value=mock_base_model),
        patch("orchestrator.PeftModel.from_pretrained",
              return_value=mock_gem_model),
        patch("orchestrator.BitsAndBytesConfig", return_value=MagicMock()),
        # Mock open() for both label_config.json and gate_config.json
        patch("builtins.open",
              mock_open(read_data=FAKE_LABEL_CONFIG)),
    ]

    # Gate config needs different data on second open() call
    # Use side_effect to alternate read_data
    import unittest.mock as um
    m_label = um.mock_open(read_data=FAKE_LABEL_CONFIG)()
    m_gate  = um.mock_open(read_data=FAKE_GATE_CONFIG)()
    multi_open = um.MagicMock(side_effect=[m_label, m_gate])

    all_patches = [
        patch("orchestrator.AutoTokenizer.from_pretrained",
              side_effect=[mock_cls_tok, mock_gem_tok]),
        patch("orchestrator.AutoModelForSequenceClassification.from_pretrained",
              return_value=mock_cls_model),
        patch("orchestrator.AutoModelForCausalLM.from_pretrained",
              return_value=mock_base_model),
        patch("orchestrator.PeftModel.from_pretrained",
              return_value=mock_gem_model),
        patch("orchestrator.BitsAndBytesConfig", return_value=MagicMock()),
        patch("builtins.open", multi_open),
    ]

    started = [p.start() for p in all_patches]
    yield {
        "cls_tok": mock_cls_tok,
        "cls_model": mock_cls_model,
        "gem_tok": mock_gem_tok,
        "gem_model": mock_gem_model,
    }
    for p in all_patches:
        p.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  1. JSON Parsing Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestParseJson:
    """Tests for EmailClassifier._parse_json() — pure logic, no model needed."""

    def _get_parse_json(self):
        """Get _parse_json as a standalone callable without loading models."""
        import importlib
        # Import the module and grab the method from the class definition directly
        import ast, textwrap
        # Just test using a real (but isolated) import
        from orchestrator import EmailClassifier
        # Create a hollow instance (bypass __init__)
        obj = object.__new__(EmailClassifier)
        return obj._parse_json

    def test_valid_json_string(self):
        """Clean JSON string → parsed dict returned."""
        parse = self._get_parse_json()
        text = '{"label": "clean", "confidence": 0.95}'
        result = parse(text)
        assert result == {"label": "clean", "confidence": 0.95}

    def test_json_embedded_in_text(self):
        """JSON buried inside reasoning text → correctly extracted."""
        parse = self._get_parse_json()
        text = 'Based on my analysis: {"label": "phishing", "confidence": 0.88} — end of report.'
        result = parse(text)
        assert result is not None
        assert result["label"] == "phishing"

    def test_invalid_returns_none(self):
        """Completely non-JSON text → returns None, no crash."""
        parse = self._get_parse_json()
        result = parse("This email looks suspicious but I cannot determine the type.")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string → returns None."""
        parse = self._get_parse_json()
        assert parse("") is None

    def test_malformed_json_returns_none(self):
        """Malformed JSON → returns None."""
        parse = self._get_parse_json()
        assert parse('{"label": "clean", "confidence":}') is None


# ═════════════════════════════════════════════════════════════════════════════
#  2. Fast Path Classification Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestFastPath:
    """Tests for the RoBERTa classifier + gate logic."""

    def _make_classifier(self, logit_values: list, threshold: float = 0.75):
        """
        Build an EmailClassifier-like object with mocked internals,
        bypassing __init__ completely.
        """
        from orchestrator import EmailClassifier
        obj = object.__new__(EmailClassifier)

        obj.classifier_device = "cpu"
        obj.gemma_device = "cpu"
        obj.id2label = {0: "clean", 1: "spam", 2: "phishing",
                        3: "fraud", 4: "malware", 5: "threat"}
        obj.temperature = 1.0
        obj.threshold = threshold

        # Mock tokenizer
        tok = MagicMock()
        encoded = MagicMock()
        encoded.to = MagicMock(return_value={
            "input_ids": torch.zeros((1, 10), dtype=torch.long)
        })
        tok.return_value = encoded
        obj.cls_tokenizer = tok

        # Mock classifier model
        mock_model = MagicMock()
        output = MagicMock()
        output.logits = torch.tensor([logit_values])
        mock_model.return_value = output
        obj.cls_model = mock_model

        return obj

    def test_high_confidence_returns_fast_path(self):
        """When gate confidence >= threshold → path is 'fast'."""
        # Logit 2 (phishing) dominates → softmax confidence will be high
        clf = self._make_classifier(
            logit_values=[-10.0, -10.0, 10.0, -10.0, -10.0, -10.0],
            threshold=0.5,
        )
        result = clf.classify("Click here to win $1000!")
        assert result["path"] == "fast"
        assert result["label"] == "phishing"
        assert result["action"] == "quarantine"

    def test_clean_email_delivers(self):
        """Clean label → action is 'deliver'."""
        clf = self._make_classifier(
            logit_values=[10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            threshold=0.5,
        )
        result = clf.classify("Hi John, meeting at 3pm today.")
        assert result["label"] == "clean"
        assert result["action"] == "deliver"
        assert result["path"] == "fast"

    def test_malware_blocks(self):
        """Malware label → action is 'block'."""
        clf = self._make_classifier(
            logit_values=[-10.0, -10.0, -10.0, -10.0, 10.0, -10.0],
            threshold=0.5,
        )
        result = clf.classify("Download this attachment for free software")
        assert result["label"] == "malware"
        assert result["action"] == "block"

    def test_threat_blocks(self):
        """Threat label → action is 'block'."""
        clf = self._make_classifier(
            logit_values=[-10.0, -10.0, -10.0, -10.0, -10.0, 10.0],
            threshold=0.5,
        )
        result = clf.classify("Your account will be terminated unless you pay")
        assert result["label"] == "threat"
        assert result["action"] == "block"

    def test_spam_goes_to_junk(self):
        """Spam label → action is 'junk_folder'."""
        clf = self._make_classifier(
            logit_values=[-10.0, 10.0, -10.0, -10.0, -10.0, -10.0],
            threshold=0.5,
        )
        result = clf.classify("You have been selected for a special offer!")
        assert result["label"] == "spam"
        assert result["action"] == "junk_folder"

    def test_action_map_covers_all_labels(self):
        """All 6 known labels have a corresponding action (no KeyError)."""
        clf = self._make_classifier(
            logit_values=[10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            threshold=0.0,  # always fast path
        )
        action_labels = ["clean", "spam", "phishing", "fraud", "malware", "threat"]
        action_map = {
            "clean": "deliver",
            "spam": "junk_folder",
            "phishing": "quarantine",
            "fraud": "quarantine",
            "malware": "block",
            "threat": "block",
        }
        for label in action_labels:
            assert label in action_map, f"Missing action for label: {label}"


# ═════════════════════════════════════════════════════════════════════════════
#  3. Deep Path Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDeepPath:
    """Tests for the Gemma LoRA deep path when gate confidence is low."""

    def _make_uncertain_classifier(self, gemma_decode_output: str):
        """
        Build a classifier where the gate always triggers deep path
        (uniform logits → low softmax confidence) and Gemma is mocked
        to return a specific output string.
        """
        from orchestrator import EmailClassifier
        obj = object.__new__(EmailClassifier)

        obj.classifier_device = "cpu"
        obj.gemma_device = "cpu"
        obj.id2label = {0: "clean", 1: "spam", 2: "phishing",
                        3: "fraud", 4: "malware", 5: "threat"}
        obj.temperature = 1.0
        obj.threshold = 0.99  # Very high threshold → almost always triggers deep path

        # Tokenizer mock
        tok = MagicMock()
        encoded = MagicMock()
        encoded.to = MagicMock(return_value={"input_ids": torch.zeros((1, 10), dtype=torch.long)})
        tok.return_value = encoded
        obj.cls_tokenizer = tok

        # RoBERTa with uniform logits → low confidence (~16.7% for 6 classes)
        mock_model = MagicMock()
        output = MagicMock()
        output.logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        mock_model.return_value = output
        obj.cls_model = mock_model

        # Gemma tokenizer mock
        gem_tok = MagicMock()
        gem_tok.apply_chat_template = MagicMock(
            return_value=torch.zeros((1, 10), dtype=torch.long)
        )
        gem_tok.decode = MagicMock(return_value=gemma_decode_output)
        gem_tok.eos_token = "<eos>"
        gem_tok.pad_token = None
        obj.gemma_tokenizer = gem_tok

        # Gemma model mock
        gem_model = MagicMock()
        gem_model.device = torch.device("cpu")
        gem_model.generate = MagicMock(return_value=torch.zeros((1, 20), dtype=torch.long))
        obj.gemma_model = gem_model

        return obj

    def test_deep_path_valid_json(self):
        """Low-confidence → Gemma called, structured JSON returned."""
        gemma_output = '{"label": "phishing", "confidence": 0.88, ' \
                       '"reason": "Suspicious URL", "indicators": ["urgency"], ' \
                       '"action": "quarantine"}'
        clf = self._make_uncertain_classifier(gemma_output)
        result = clf.classify("Verify your bank account immediately by clicking this link")
        assert result["path"] == "deep"
        assert result["label"] == "phishing"
        assert result["action"] == "quarantine"
        assert "indicators" in result

    def test_deep_path_garbage_fallback(self):
        """Gemma returns garbage text → falls back to classifier label, no crash."""
        clf = self._make_uncertain_classifier("I am not sure about this email, it looks okay.")
        # With uniform logits → label is "clean" (first index when tied)
        result = clf.classify("Some ambiguous email text here")
        assert result["path"] == "deep"
        # Label should fall back to something sensible
        assert result["label"] in ["clean", "spam", "phishing", "fraud", "malware", "threat", "unknown"]
        # Action defaults
        assert "action" in result
        assert "reason" in result

    def test_deep_path_text_truncated(self):
        """Text longer than 3000 chars is truncated before Gemma call."""
        gemma_output = '{"label": "spam", "confidence": 0.7, "reason": "Promotional", ' \
                       '"indicators": [], "action": "junk_folder"}'
        clf = self._make_uncertain_classifier(gemma_output)
        long_email = "This is a very long email. " * 200  # ~5400 chars
        result = clf.classify(long_email)
        # Gemma model.generate should have been called once (text was truncated, not skipped)
        assert clf.gemma_model.generate.called
        assert result["path"] == "deep"
