"""
main.py
=======
Gmail Email Agent — polls unread emails from Gmail, sends them to the
MCP Model Server for classification, applies Gmail labels, and logs
results to MongoDB.

Architecture:
    Gmail API → Extract email → POST to MCP Server → Apply label → MongoDB

Prerequisites:
    1. MCP Model Server running on port 8001:
       CUDA_VISIBLE_DEVICES=0 uvicorn jenkins.model_server:app --port 8001
    2. MongoDB running on port 27017
    3. Gmail OAuth token.json in this directory

Usage:
    uvicorn jenkins.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import re
from email.utils import parseaddr

import httpx
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# ── MCP Server Config ─────────────────────────────────────────────────────────
MCP_SERVER_URL = "http://localhost:8011"

# ── Database Schema ───────────────────────────────────────────────────────────
class EmailRecord(BaseModel):
    message_id: str
    subject: str = ""
    snippet: str
    sender: str = ""
    label_applied: str
    confidence: float = 0.0
    reason: str = ""
    indicators: list = []
    action: str = ""
    inference_path: str = "fast"
    latency_ms: float = 0.0
    status: str = "processed"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo_dict(self):
        return self.model_dump()

# ── Gmail API Setup ───────────────────────────────────────────────────────────
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    """Authenticates and returns the Gmail API service."""
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    return build('gmail', 'v1', credentials=creds)

def setup_gmail_labels(service):
    """Checks if custom labels exist in Gmail. Creates them if missing."""
    required_labels = ["clean", "phishing", "malware", "threat", "fraud"]
    label_mapping = {"spam": "SPAM"}

    results = service.users().labels().list(userId='me').execute()
    existing_labels = results.get('labels', [])
    existing_dict = {label['name'].lower(): label['id'] for label in existing_labels}

    for label_name in required_labels:
        if label_name in existing_dict:
            print(f"✅ Label '{label_name}' found. ID: {existing_dict[label_name]}")
            label_mapping[label_name] = existing_dict[label_name]
        else:
            print(f"⚠️ Label '{label_name}' not found. Creating it now...")
            label_metadata = {
                'name': label_name.capitalize(),
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            created_label = service.users().labels().create(
                userId='me', body=label_metadata
            ).execute()
            print(f"✨ Created '{label_name}' with ID: {created_label['id']}")
            label_mapping[label_name] = created_label['id']

    return label_mapping


# ── Email Body Extraction ─────────────────────────────────────────────────────
def extract_email_content(message: dict) -> dict:
    """Extract subject, sender, and full body from a Gmail message."""
    headers = message.get("payload", {}).get("headers", [])
    subject = ""
    sender = ""
    for h in headers:
        name = h.get("name", "").lower()
        if name == "subject":
            subject = h.get("value", "")
        elif name == "from":
            sender = h.get("value", "")

    # Extract body from MIME parts
    body = _extract_body(message.get("payload", {}))

    # Clean HTML tags if present
    body = re.sub(r'<[^>]+>', ' ', body)
    body = re.sub(r'\s+', ' ', body).strip()

    # Combine subject + body for classification
    full_text = f"Subject: {subject}\n\n{body}" if subject else body

    return {
        "subject": subject,
        "sender": parseaddr(sender)[1] if sender else "",
        "body": body,
        "full_text": full_text,
        "snippet": message.get("snippet", ""),
    }


def _extract_body(payload: dict) -> str:
    """Recursively extract text body from MIME payload."""
    body = ""

    # Direct body data
    if payload.get("body", {}).get("data"):
        try:
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        except Exception:
            pass

    # Multipart — recurse into parts
    for part in payload.get("parts", []):
        mime = part.get("mimeType", "")
        if mime == "text/plain":
            if part.get("body", {}).get("data"):
                try:
                    body = base64.urlsafe_b64decode(
                        part["body"]["data"]
                    ).decode("utf-8", errors="replace")
                    break  # prefer plain text
                except Exception:
                    pass
        elif mime == "text/html" and not body:
            if part.get("body", {}).get("data"):
                try:
                    body = base64.urlsafe_b64decode(
                        part["body"]["data"]
                    ).decode("utf-8", errors="replace")
                except Exception:
                    pass
        elif mime.startswith("multipart/"):
            body = _extract_body(part)
            if body:
                break

    return body


# ── MCP Server Communication ─────────────────────────────────────────────────
async def classify_via_mcp(email_text: str) -> dict:
    """Send email text to MCP Model Server for classification."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{MCP_SERVER_URL}/classify",
                json={"email_text": email_text}
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError:
            print("⚠️ MCP Server unreachable — falling back to 'review'")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "reason": "MCP server unavailable",
                "indicators": [],
                "action": "review",
                "path": "fallback",
                "latency_ms": 0,
            }
        except Exception as e:
            print(f"⚠️ MCP classification error: {e}")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "reason": str(e),
                "indicators": [],
                "action": "review",
                "path": "error",
                "latency_ms": 0,
            }


# ── Background Polling Loop ──────────────────────────────────────────────────
async def poll_gmail_inbox(db):
    print("Initializing Gmail Service...")
    service = get_gmail_service()

    print("Checking/Creating required Gmail Labels...")
    dynamic_label_mapping = setup_gmail_labels(service)
    print("Label mapping ready!", dynamic_label_mapping)

    while True:
        try:
            results = service.users().messages().list(
                userId='me', q='is:unread in:inbox'
            ).execute()
            messages = results.get('messages', [])

            for msg_ref in messages:
                msg_id = msg_ref['id']
                message = service.users().messages().get(
                    userId='me', id=msg_id, format='full'
                ).execute()

                # Extract full email content
                email_data = extract_email_content(message)
                email_text = email_data["full_text"]

                # Truncate very long emails
                if len(email_text) > 5000:
                    email_text = email_text[:5000] + " …"

                # ──────────────────────────────────────────────────────
                # CLASSIFY VIA MCP SERVER
                # ──────────────────────────────────────────────────────
                result = await classify_via_mcp(email_text)
                classification = result.get("label", "unknown")

                print(f"📧 [{result.get('path','?')}] {email_data['subject'][:50]} → "
                      f"{classification} ({result.get('confidence',0):.1%})")

                # Apply Gmail label
                label_id_to_apply = dynamic_label_mapping.get(classification.lower())

                if label_id_to_apply:
                    labels_to_remove = ['UNREAD']

                    if classification.lower() != "clean":
                        labels_to_remove.append('INBOX')

                    service.users().messages().modify(
                        userId='me',
                        id=msg_id,
                        body={
                            'addLabelIds': [label_id_to_apply],
                            'removeLabelIds': labels_to_remove
                        }
                    ).execute()

                    # Save to MongoDB
                    try:
                        new_record = EmailRecord(
                            message_id=msg_id,
                            subject=email_data["subject"],
                            snippet=email_data["snippet"],
                            sender=email_data["sender"],
                            label_applied=classification,
                            confidence=result.get("confidence", 0.0),
                            reason=result.get("reason", ""),
                            indicators=result.get("indicators", []),
                            action=result.get("action", ""),
                            inference_path=result.get("path", "fast"),
                            latency_ms=result.get("latency_ms", 0),
                        )
                        db_result = await db.emails.insert_one(new_record.to_mongo_dict())
                        print(f"  ✅ Saved to DB: {db_result.inserted_id}")
                    except Exception as db_error:
                        print(f"  ❌ Failed to save to MongoDB: {db_error}")

        except Exception as e:
            print(f"Error checking email: {e}")

        await asyncio.sleep(60)


# ── App Initialization & Routing ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to MongoDB on port 27017...")

    app.mongodb_client = AsyncIOMotorClient(
        "mongodb://test_1:sentinelMongo_test123@localhost:27017/jenkins?authSource=jenkins",
        serverSelectionTimeoutMS=5000
    )
    app.db = app.mongodb_client.jenkins

    try:
        await app.mongodb_client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
    except Exception as e:
        print(f"❌ CRITICAL: Could not connect to MongoDB. Error: {e}")

    asyncio.create_task(poll_gmail_inbox(app.db))
    yield

    print("Closing MongoDB connection...")
    app.mongodb_client.close()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Check both Gmail agent and MCP server health."""
    mcp_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MCP_SERVER_URL}/health")
            mcp_status = resp.json().get("status", "unknown")
    except Exception:
        mcp_status = "unreachable"

    return {
        "agent": "running",
        "mcp_server": mcp_status,
        "polling": "active",
    }

@app.get("/api/logs")
async def get_email_logs():
    """Fetches the last 20 processed emails from the database."""
    cursor = app.db.emails.find().sort("created_at", -1).limit(20)
    records = await cursor.to_list(length=20)

    for record in records:
        record["_id"] = str(record["_id"])

    return {"total_returned": len(records), "logs": records}

@app.post("/api/classify")
async def manual_classify(email_text: str):
    """Manually classify an email (for testing)."""
    result = await classify_via_mcp(email_text)
    return result