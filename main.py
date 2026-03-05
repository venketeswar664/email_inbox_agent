import asyncio
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# --- Database Schema ---
class EmailRecord(BaseModel):
    message_id: str
    snippet: str
    label_applied: str
    status: str = "processed"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo_dict(self):
        return self.model_dump()

# --- Gmail API Setup ---
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def get_gmail_service():
    """Authenticates and returns the Gmail API service."""
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    return build('gmail', 'v1', credentials=creds)

def setup_gmail_labels(service):
    """Checks if custom labels exist in Gmail. Creates them if missing."""
    required_labels = ["clean", "phishing", "malware", "threats"]
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
            created_label = service.users().labels().create(userId='me', body=label_metadata).execute()
            print(f"✨ Created '{label_name}' with ID: {created_label['id']}")
            label_mapping[label_name] = created_label['id']

    return label_mapping

# --- Background Polling Loop ---
async def poll_gmail_inbox(db):
    print("Initializing Gmail Service...")
    service = get_gmail_service()
    
    print("Checking/Creating required Gmail Labels...")
    dynamic_label_mapping = setup_gmail_labels(service)
    print("Label mapping ready!", dynamic_label_mapping)

    while True:
        try:
            results = service.users().messages().list(userId='me', q='is:unread in:inbox').execute()
            messages = results.get('messages', [])

            for msg_ref in messages:
                msg_id = msg_ref['id']
                message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                email_snippet = message.get('snippet', '')

                # ---------------------------------------------------------
                # CALL YOUR ORCHESTRATOR HERE
                # ---------------------------------------------------------
                classification = "phishing"  # Hardcoded for testing. 

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

                    # --- SCHEMA VALIDATION AND DB INSERT ---
                    try:
                        new_record = EmailRecord(
                            message_id=msg_id,
                            snippet=email_snippet,
                            label_applied=classification
                        )
                        # Await is critical here to ensure the write finishes
                        result = await db.emails.insert_one(new_record.to_mongo_dict())
                        print(f"✅ Saved to DB with Mongo ID: {result.inserted_id}")
                    except Exception as db_error:
                         print(f"❌ Failed to save to MongoDB: {db_error}")

        except Exception as e:
            print(f"Error checking email: {e}")
        
        await asyncio.sleep(60)

# --- App Initialization & Routing ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to MongoDB on port 27017...")
    
    # 1. Connect and authenticate to the jenkins database
    app.mongodb_client = AsyncIOMotorClient("mongodb://test_1:sentinelMongo_test123@localhost:27017/jenkins?authSource=jenkins", serverSelectionTimeoutMS=5000)
    
    # 2. Tell the app to USE the jenkins database for storing emails
    app.db = app.mongodb_client.jenkins
    
    try:
        # Force a ping to verify the connection is alive
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
def health_check():
    return {"status": "Agent is running, connected to MongoDB, and polling Gmail."}

@app.get("/api/logs")
async def get_email_logs():
    """Fetches the last 10 processed emails from the database"""
    # Fetch data and sort by newest first
    cursor = app.db.emails.find().sort("created_at", -1).limit(10)
    records = await cursor.to_list(length=10)
    
    # Convert MongoDB's internal ObjectId to a string so FastAPI can return it as JSON
    for record in records:
        record["_id"] = str(record["_id"])
        
    return {"total_returned": len(records), "logs": records}