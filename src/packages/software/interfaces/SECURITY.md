# Security Policy - Interfaces Package

## Overview

The Interfaces package provides all user-facing interfaces including REST APIs, CLI tools, web applications, and client SDKs. As the primary entry point for users and external systems, security is critical to protect against attacks, ensure data privacy, and maintain system integrity.

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | End of Life    |
| ------- | ------------------ | -------------- |
| 2.x.x   | :white_check_mark: | -              |
| 1.9.x   | :white_check_mark: | 2025-06-01     |
| 1.8.x   | :warning:          | 2024-12-31     |
| < 1.8   | :x:                | Ended          |

## Security Model

### Interface Security Domains

Our security model addresses these critical areas for user-facing interfaces:

**1. API Security**
- Authentication and authorization
- Input validation and sanitization
- Rate limiting and DoS protection
- CORS and security headers

**2. Web Application Security**
- Cross-Site Scripting (XSS) prevention
- Cross-Site Request Forgery (CSRF) protection
- Content Security Policy (CSP)
- Secure session management

**3. CLI Security**
- Command injection prevention
- Secure credential handling
- Safe file operations
- Process isolation

**4. SDK Security**
- Secure communication protocols
- API key and token management
- Certificate validation
- Data transmission security

## Threat Model

### High-Risk Scenarios

**API Attacks**
- SQL injection through API parameters
- NoSQL injection in query parameters
- Authentication bypass attempts
- Rate limiting circumvention
- API enumeration and data extraction

**Web Application Attacks**
- Cross-Site Scripting (XSS) attacks
- Cross-Site Request Forgery (CSRF)
- Session hijacking and fixation
- File upload vulnerabilities
- DOM-based attacks

**Command Line Interface Attacks**
- Command injection through user inputs
- Path traversal in file operations
- Credential theft from command history
- Process elevation attacks
- Environment variable manipulation

**Client SDK Attacks**
- Man-in-the-middle attacks
- Certificate pinning bypass
- API key extraction from client code
- Insecure storage of credentials
- Network traffic interception

## Security Features

### API Security

**Authentication and Authorization**
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthenticationService:
    """Secure authentication service for API access."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create secure JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        # Add security claims
        to_encode.update({
            "iss": "pynomaly-api",  # Issuer
            "aud": "pynomaly-users",  # Audience
            "jti": str(uuid4())  # JWT ID for revocation
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token with security checks."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True
                }
            )
            
            # Additional security validations
            if payload.get("iss") != "pynomaly-api":
                raise JWTError("Invalid issuer")
            
            if payload.get("aud") != "pynomaly-users":
                raise JWTError("Invalid audience")
            
            return payload
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthenticationService = Depends()
) -> dict:
    """Secure user authentication dependency."""
    payload = auth_service.verify_token(credentials.credentials)
    user_id = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Additional user validation
    user = await get_user_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive",
        )
    
    return user
```

**Input Validation and Sanitization**
```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import re
import html

class SecureDetectionRequest(BaseModel):
    """Secure request model with comprehensive validation."""
    
    dataset_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        regex="^[a-zA-Z0-9_-]+$",
        description="Dataset identifier"
    )
    algorithm: str = Field(
        ...,
        description="Detection algorithm"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default={},
        description="Algorithm parameters"
    )
    
    @validator('dataset_id')
    def validate_dataset_id(cls, v):
        """Validate dataset ID to prevent injection attacks."""
        # Check for SQL injection patterns
        sql_patterns = [
            r"['\"`;]",  # SQL metacharacters
            r"\b(union|select|insert|update|delete|drop|create)\b",  # SQL keywords
            r"--",  # SQL comments
            r"/\*.*\*/"  # SQL block comments
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid characters in dataset_id")
        
        return v
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        """Validate algorithm name against whitelist."""
        allowed_algorithms = {
            'isolation_forest', 'lof', 'one_class_svm', 
            'autoencoder', 'statistical'
        }
        
        if v not in allowed_algorithms:
            raise ValueError(f"Algorithm must be one of: {allowed_algorithms}")
        
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v):
        """Validate parameters to prevent code injection."""
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Limit parameter count and size
        if len(v) > 50:
            raise ValueError("Too many parameters")
        
        for key, value in v.items():
            # Validate parameter keys
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid parameter name: {key}")
            
            # Validate parameter values
            if isinstance(value, str):
                if len(value) > 1000:
                    raise ValueError(f"Parameter {key} value too long")
                
                # HTML escape string values
                v[key] = html.escape(value)
            
            elif isinstance(value, (int, float)):
                # Check for reasonable numeric ranges
                if abs(value) > 1e10:
                    raise ValueError(f"Parameter {key} value out of range")
        
        return v
```

**Rate Limiting and DoS Protection**
```python
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
from typing import Optional

# Rate limiter with Redis backend
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    storage_uri="redis://localhost:6379"
)

class AdvancedRateLimiter:
    """Advanced rate limiting with user-based and endpoint-specific limits."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def check_rate_limit(
        self, 
        request: Request,
        endpoint: str,
        user_id: Optional[str] = None,
        limit_per_minute: int = 60
    ) -> bool:
        """Check rate limit with multiple strategies."""
        
        # Get client identifier
        client_id = user_id or get_remote_address(request)
        
        # Create rate limit keys
        keys = [
            f"rate_limit:global:{client_id}",
            f"rate_limit:endpoint:{endpoint}:{client_id}",
            f"rate_limit:user:{user_id}" if user_id else None
        ]
        keys = [k for k in keys if k is not None]
        
        # Check all rate limits
        for key in keys:
            current_requests = await self.redis.get(key)
            
            if current_requests is None:
                # First request in window
                await self.redis.setex(key, 60, 1)
            else:
                current_count = int(current_requests)
                if current_count >= limit_per_minute:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded for {endpoint}",
                        headers={"Retry-After": "60"}
                    )
                
                await self.redis.incr(key)
        
        return True

# Apply rate limiting to endpoints
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Apply different limits based on endpoint
    endpoint_limits = {
        "/api/v1/detection/detect": 10,  # Expensive operations
        "/api/v1/datasets": 30,  # Dataset operations
        "/api/v1/models": 20,  # Model operations
    }
    
    limit = endpoint_limits.get(request.url.path, 60)  # Default limit
    
    rate_limiter = AdvancedRateLimiter(redis_client)
    await rate_limiter.check_rate_limit(
        request=request,
        endpoint=request.url.path,
        limit_per_minute=limit
    )
    
    return await call_next(request)
```

### Web Application Security

**XSS Prevention and CSP**
```python
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import html
import bleach

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure secure Jinja2 environment
templates.env.finalize = lambda x: x if x is not None else ''
templates.env.autoescape = True

class SecurityMiddleware:
    """Security middleware for web application protection."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Apply security headers and protections."""
        response = await call_next(request)
        
        # Content Security Policy
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'"
        )
        response.headers["Content-Security-Policy"] = csp_policy
        
        # Additional security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response

# CSRF Protection
from fastapi_csrf_protect import CsrfProtect
from fastapi_csrf_protect.exceptions import CsrfProtectError

class CsrfSettings(BaseModel):
    secret_key: str = "your-secret-key"
    
@CsrfProtect.load_config
def get_csrf_config():
    return CsrfSettings()

# Secure template rendering
def secure_render_template(
    name: str,
    context: dict,
    request: Request
) -> Response:
    """Render template with security protections."""
    
    # Sanitize all string values in context
    sanitized_context = {}
    for key, value in context.items():
        if isinstance(value, str):
            # Use bleach to clean HTML
            sanitized_context[key] = bleach.clean(
                value,
                tags=['b', 'i', 'u', 'em', 'strong', 'p', 'br'],
                attributes={},
                strip=True
            )
        else:
            sanitized_context[key] = value
    
    # Add CSRF token
    csrf_token = CsrfProtect().generate_csrf()
    sanitized_context['csrf_token'] = csrf_token
    
    return templates.TemplateResponse(
        name,
        {"request": request, **sanitized_context}
    )
```

### CLI Security

**Command Injection Prevention**
```python
import subprocess
import shlex
from pathlib import Path
from typing import List, Optional
import os

class SecureCLIHandler:
    """Secure CLI command handler with injection prevention."""
    
    def __init__(self):
        self.allowed_commands = {
            'detect', 'train', 'evaluate', 'export', 'import'
        }
        self.safe_file_extensions = {'.csv', '.json', '.parquet', '.txt'}
    
    def validate_command(self, command: str) -> bool:
        """Validate command against whitelist."""
        if command not in self.allowed_commands:
            raise ValueError(f"Command '{command}' not allowed")
        return True
    
    def sanitize_file_path(self, file_path: str) -> Path:
        """Sanitize and validate file path."""
        # Convert to Path object for safe handling
        path = Path(file_path).resolve()
        
        # Check file extension
        if path.suffix.lower() not in self.safe_file_extensions:
            raise ValueError(f"File type '{path.suffix}' not allowed")
        
        # Prevent path traversal
        try:
            # Ensure path is within allowed directories
            cwd = Path.cwd().resolve()
            path.relative_to(cwd)
        except ValueError:
            raise ValueError("Path outside allowed directory")
        
        # Check for dangerous characters
        dangerous_chars = ['|', '&', ';', '$', '`', '(', ')', '<', '>']
        if any(char in str(path) for char in dangerous_chars):
            raise ValueError("Invalid characters in file path")
        
        return path
    
    def execute_safe_subprocess(
        self,
        command: List[str],
        timeout: int = 30,
        cwd: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """Execute subprocess with security restrictions."""
        
        # Validate all command arguments
        safe_args = []
        for arg in command:
            # Prevent command injection
            if any(char in arg for char in ['|', '&', ';', '$', '`']):
                raise ValueError(f"Unsafe argument: {arg}")
            
            # Shell escape the argument
            safe_args.append(shlex.quote(str(arg)))
        
        # Set secure environment
        env = os.environ.copy()
        # Remove potentially dangerous environment variables
        dangerous_env_vars = ['LD_PRELOAD', 'DYLD_INSERT_LIBRARIES']
        for var in dangerous_env_vars:
            env.pop(var, None)
        
        try:
            result = subprocess.run(
                safe_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
                shell=False  # Never use shell=True
            )
            return result
            
        except subprocess.TimeoutExpired:
            raise ValueError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise ValueError(f"Command execution failed: {str(e)}")

# Secure file operations
class SecureFileHandler:
    """Secure file handling for CLI operations."""
    
    def __init__(self, max_file_size: int = 100 * 1024 * 1024):  # 100MB
        self.max_file_size = max_file_size
        self.temp_dir = Path.cwd() / "temp"
        self.temp_dir.mkdir(exist_ok=True)
    
    def read_file_safely(self, file_path: Path) -> str:
        """Read file with security checks."""
        # Check file size
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {file_path}")
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            raise ValueError(f"No read permission: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Additional content validation
            if len(content) > self.max_file_size:
                raise ValueError("File content too large")
            
            return content
            
        except UnicodeDecodeError:
            raise ValueError("File is not valid UTF-8 text")
        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}")
    
    def write_file_safely(self, file_path: Path, content: str) -> None:
        """Write file with security checks."""
        # Validate content size
        if len(content) > self.max_file_size:
            raise ValueError("Content too large")
        
        # Ensure directory exists and is writable
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        temp_file = self.temp_dir / f"temp_{os.getpid()}_{file_path.name}"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atomic move to final location
            temp_file.replace(file_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise ValueError(f"Failed to write file: {str(e)}")
```

### SDK Security

**Secure Client Implementation**
```python
import ssl
import certifi
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import Optional, Dict, Any

class SecureAPIClient:
    """Secure API client with comprehensive security features."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Create secure session
        self.session = requests.Session()
        
        # Configure SSL/TLS
        if verify_ssl:
            # Use system certificate store
            self.session.verify = certifi.where()
            
            # Create secure SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            # Disable weak ciphers
            ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        else:
            self.session.verify = False
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set secure headers
        self.session.headers.update({
            'User-Agent': 'Pynomaly-SDK/1.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Add API key if provided
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None
    ) -> requests.Response:
        """Make secure HTTP request with validation."""
        
        # Validate endpoint
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        # Construct full URL
        url = f"{self.base_url}{endpoint}"
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            raise ValueError("Invalid URL scheme")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            
            # Validate response
            if response.status_code >= 400:
                self._handle_error_response(response)
            
            return response
            
        except requests.exceptions.SSLError as e:
            raise SecurityError(f"SSL verification failed: {str(e)}")
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Connection failed: {str(e)}")
    
    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle API error responses securely."""
        try:
            error_data = response.json()
            error_message = error_data.get('detail', 'Unknown error')
        except:
            error_message = f"HTTP {response.status_code} error"
        
        # Don't expose sensitive information
        safe_message = self._sanitize_error_message(error_message)
        
        if response.status_code == 401:
            raise AuthenticationError(safe_message)
        elif response.status_code == 403:
            raise AuthorizationError(safe_message)
        elif response.status_code == 429:
            raise RateLimitError(safe_message)
        else:
            raise APIError(safe_message)
    
    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error messages to prevent information disclosure."""
        # Remove potential sensitive information
        import re
        
        # Remove file paths
        message = re.sub(r'/[^\s]*', '[PATH]', message)
        
        # Remove IP addresses
        message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', message)
        
        # Remove UUIDs and tokens
        message = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '[UUID]', message)
        message = re.sub(r'[A-Za-z0-9+/]{20,}={0,2}', '[TOKEN]', message)
        
        return message
```

## Security Best Practices

### Development

**Secure Development Lifecycle**
- Security reviews for all interface changes
- Static analysis scanning for vulnerabilities
- Dependency vulnerability scanning
- Regular security testing and penetration testing
- Secure coding training for developers

**Input Validation**
- Validate all user inputs at interface boundaries
- Use allow-lists rather than deny-lists
- Implement proper type checking and range validation
- Sanitize outputs to prevent injection attacks
- Use parameterized queries for database operations

**Authentication and Authorization**
- Implement strong authentication mechanisms
- Use secure session management
- Apply principle of least privilege
- Regular token rotation and expiration
- Multi-factor authentication support

### Deployment

**Production Security**
- Use HTTPS for all communications
- Implement proper CORS policies
- Configure security headers
- Regular security updates and patches
- Monitor for security incidents

**Infrastructure Security**
- Secure server configurations
- Network segmentation and firewalls
- Regular security assessments
- Backup and disaster recovery plans
- Incident response procedures

## Vulnerability Reporting

### Reporting Process

Interface vulnerabilities can affect all users and require immediate attention.

**1. Critical Interface Vulnerabilities**
- Authentication/authorization bypasses
- XSS, CSRF, or injection vulnerabilities
- API security flaws
- Data exposure through interfaces

**2. Contact Security Team**
- Email: interfaces-security@yourorg.com
- PGP Key: [Provide interfaces security PGP key]
- Include "Interfaces Security Vulnerability" in the subject line

**3. Provide Detailed Information**
```
Subject: Interfaces Security Vulnerability - [Brief Description]

Vulnerability Details:
- Interface component: [e.g., REST API, web app, CLI, SDK]
- Vulnerability type: [e.g., XSS, injection, authentication bypass]
- Severity level: [Critical/High/Medium/Low]
- Attack vector: [How the vulnerability can be exploited]
- User impact: [What users could be affected]
- Reproduction steps: [Detailed steps to reproduce]
- Proof of concept: [If available, but avoid causing harm]
- Suggested fix: [If you have recommendations]

Environment Information:
- Interfaces package version: [Version number]
- Browser/client information: [If applicable]
- Operating system: [OS and version]
- Additional dependencies: [FastAPI, etc.]
```

### Response Timeline

**Critical Interface Vulnerabilities**
- **Acknowledgment**: Within 1 hour
- **Initial Assessment**: Within 4 hours
- **Emergency Response**: Within 8 hours if actively exploited
- **Resolution Timeline**: 24-48 hours depending on complexity

**High/Medium Severity**
- **Acknowledgment**: Within 4 hours
- **Initial Assessment**: Within 24 hours
- **Detailed Analysis**: Within 72 hours
- **Resolution Timeline**: 1-2 weeks depending on impact

## Contact Information

**Interfaces Security Team**
- Email: interfaces-security@yourorg.com
- Emergency Phone: [Emergency contact for critical interface vulnerabilities]
- PGP Key: [Interfaces security PGP key fingerprint]

**Escalation Contacts**
- Interfaces Package Maintainer: [Contact information]
- Security Architect: [Contact information]
- User Experience Lead: [Contact information]

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Next Review**: March 2025