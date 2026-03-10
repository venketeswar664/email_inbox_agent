"""
conftest.py
===========
Shared pytest configuration and fixtures for the Email Inbox Agent test suite.

This file is automatically loaded by pytest before any test files.
Fixtures defined here are available in ALL test files without importing.

Key responsibilities:
  1. Set PYTHONPATH so tests can import from the repo root (main, orchestrator, etc.)
  2. Provide shared mock objects used across multiple test files
  3. Configure asyncio event loop scope
"""

import sys
import os
import pytest

# ── Path Setup ────────────────────────────────────────────────────────────────
# Add the repo root to sys.path so pytest can find main.py, orchestrator.py, etc.
# This is the key fix that lets tests run from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Shared Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fake_label_config():
    """6-class label map used by orchestrator tests."""
    return {
        0: "clean",
        1: "spam",
        2: "phishing",
        3: "fraud",
        4: "malware",
        5: "threat",
    }


@pytest.fixture(scope="session")
def sample_emails():
    """
    Dictionary of representative email texts for testing.
    Keys: descriptive name
    Values: email string (subject + body)
    """
    return {
        "clean": (
            "Subject: Team lunch today at 12:30pm\n\n"
            "Hi all, just a reminder that we have a team lunch today at 12:30 "
            "in the main cafeteria. Looking forward to seeing everyone!"
        ),
        "phishing": (
            "Subject: URGENT: Your account has been compromised!\n\n"
            "Dear valued customer, we have detected suspicious activity on your account. "
            "Please verify your identity immediately by clicking: http://secure-login-update.xyz/verify "
            "Failure to act within 24 hours will result in your account being permanently suspended."
        ),
        "spam": (
            "Subject: You've been selected for an exclusive offer!\n\n"
            "Congratulations! You are our LUCKY WINNER this month. "
            "Click here to claim your FREE iPhone 15 Pro. Limited time offer!"
        ),
        "malware": (
            "Subject: Your invoice is attached\n\n"
            "Please find your invoice in the attached file. "
            "Open the document to see your payment details. invoice_march_2026.exe"
        ),
        "fraud": (
            "Subject: Wire transfer request — urgent\n\n"
            "Hi, I am the CEO and I need you to urgently wire $50,000 to the following account. "
            "This is confidential. Do not tell anyone. Bank: XYZ, Account: 123456789"
        ),
        "threat": (
            "Subject: Final warning before legal action\n\n"
            "You have 48 hours to pay $500 in Bitcoin or we will expose your private information "
            "to all your contacts and your employer. This is your only warning."
        ),
    }
