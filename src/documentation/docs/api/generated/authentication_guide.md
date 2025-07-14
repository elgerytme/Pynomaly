# Authentication Guide

## Overview

Pynomaly API supports multiple authentication methods:
- JWT Token Authentication (recommended for user sessions)
- API Key Authentication (recommended for server-to-server)
- Multi-Factor Authentication (MFA) for enhanced security

## JWT Token Authentication

### Flow

1. **Login** with credentials
2. **Receive** JWT access token and refresh token
3. **Use** access token in API requests
4. **Refresh** token when expired

### Implementation

```python
import requests
import json
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def login(self, username: str, password: str) -> bool:
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])
            return True
        return False
    
    def refresh_access_token(self) -> bool:
        if not self.refresh_token:
            return False
        
        response = requests.post(
            f"{self.base_url}/api/v1/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])
            return True
        return False
    
    def get_auth_headers(self) -> dict:
        # Auto-refresh if token is expired
        if self.token_expires_at and datetime.now() >= self.token_expires_at:
            self.refresh_access_token()
        
        return {"Authorization": f"Bearer {self.access_token}"}
```

## API Key Authentication

### Creation

```bash
# Create API key (requires admin privileges)
curl -X POST http://localhost:8000/api/v1/api-keys \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production API Key",
    "scopes": ["detection:read", "detection:write", "health:read"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

### Usage

```python
import requests

class APIKeyClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def make_request(self, method: str, endpoint: str, data: dict = None):
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, json=data)
        return response.json()
```

## Multi-Factor Authentication (MFA)

### TOTP (Time-based One-Time Password)

```python
import pyotp
import qrcode
from io import BytesIO
import base64

class TOTPManager:
    def __init__(self, api_client):
        self.api_client = api_client
    
    def setup_totp(self, app_name: str = "Pynomaly") -> dict:
        response = self.api_client.post("/api/v1/mfa/totp/setup", {
            "app_name": app_name,
            "issuer": "Pynomaly Security"
        })
        return response.json()
    
    def verify_totp(self, totp_code: str) -> bool:
        response = self.api_client.post("/api/v1/mfa/totp/verify", {
            "totp_code": totp_code
        })
        return response.status_code == 200
    
    def generate_qr_code(self, totp_uri: str) -> str:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
```

### SMS Authentication

```python
class SMSAuth:
    def __init__(self, api_client):
        self.api_client = api_client
    
    def setup_sms(self, phone_number: str) -> bool:
        response = self.api_client.post("/api/v1/mfa/sms/setup", {
            "phone_number": phone_number
        })
        return response.status_code == 200
    
    def verify_sms(self, sms_code: str) -> bool:
        response = self.api_client.post("/api/v1/mfa/sms/verify", {
            "sms_code": sms_code
        })
        return response.status_code == 200
```

## Security Best Practices

1. **Token Storage**: Store tokens securely (not in localStorage for web apps)
2. **Token Rotation**: Implement automatic token refresh
3. **API Key Management**: Rotate API keys regularly
4. **MFA Enforcement**: Enable MFA for all admin accounts
5. **Secure Transmission**: Always use HTTPS in production
6. **Rate Limiting**: Implement client-side rate limiting
7. **Error Handling**: Handle authentication errors gracefully

## Troubleshooting

### Common Issues

1. **Token Expired**: Implement automatic refresh
2. **Invalid Credentials**: Check username/password
3. **MFA Required**: Complete MFA setup
4. **Rate Limited**: Implement backoff strategy
5. **Invalid API Key**: Check key validity and scopes

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Use debug headers
headers = {
    "X-Debug": "true",
    "X-Request-ID": "unique-request-id"
}
```
