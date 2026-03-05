import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# We use 'modify' so the script can read emails and change their labels.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def generate_token():
    creds = None
    
    # Check if we already have a token
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    # If there are no valid credentials, force the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("No token found. Opening browser for authorization...")
            # This looks for the credentials.json you downloaded from Google Cloud
            flow = InstalledAppFlow.from_client_secrets_file('/home/venketeswar/Pictures/jenkins/client_secret_1095892884745-h9g8tp71d1a6rk247o5bco89fortakgq.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        # Save the credentials for all future headless runs (like in Docker/Jenkins)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            print("\n✅ SUCCESS: token.json has been generated!")

if __name__ == '__main__':
    generate_token()