import asyncio
import time
import json
import jwt
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import aiohttp
import aioredis
from prometheus_client import Counter, Histogram, Gauge
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class AuthenticationMethod(Enum):
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    MUTUAL_TLS = "mutual_tls"

class ProtocolType(Enum):
    REST = "rest"
    GRAPHQL = "graphql"
    SOAP = "soap"
    GRPC = "grpc"
    WEBSOCKET = "websocket"

class TransformationType(Enum):
    JSON_TO_XML = "json_to_xml"
    XML_TO_JSON = "xml_to_json"
    FIELD_MAPPING = "field_mapping"
    DATA_ENRICHMENT = "data_enrichment"
    PROTOCOL_ADAPTATION = "protocol_adaptation"

@dataclass
class APIEndpoint:
    id: str
    name: str
    path: str
    method: str
    protocol: ProtocolType
    upstream_url: str
    authentication: AuthenticationMethod
    rate_limit: int
    timeout_seconds: int
    retry_attempts: int
    circuit_breaker_enabled: bool
    transformation_rules: List[Dict[str, Any]]
    validation_schema: Optional[Dict[str, Any]] = None
    caching_enabled: bool = False
    cache_ttl_seconds: int = 300
    monitoring_enabled: bool = True

@dataclass
class RateLimitRule:
    client_id: str
    endpoint_id: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int

@dataclass
class TransformationRule:
    rule_id: str
    transformation_type: TransformationType
    source_format: str
    target_format: str
    field_mappings: Dict[str, str]
    enrichment_sources: List[str]
    validation_rules: List[Dict[str, Any]]

@dataclass
class CircuitBreakerState:
    endpoint_id: str
    state: str  # CLOSED, OPEN, HALF_OPEN
    failure_count: int
    last_failure_time: Optional[datetime]
    next_attempt_time: Optional[datetime]
    success_threshold: int
    failure_threshold: int
    timeout_seconds: int

class EnterpriseAPIGateway:
    def __init__(self):
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.rate_limits: Dict[str, RateLimitRule] = {}
        self.transformation_rules: Dict[str, TransformationRule] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Metrics
        self.request_counter = Counter('api_gateway_requests_total', 
                                     'Total API requests', 
                                     ['endpoint', 'method', 'status'])
        self.request_duration = Histogram('api_gateway_request_duration_seconds',
                                        'Request duration',
                                        ['endpoint', 'method'])
        self.rate_limit_exceeded = Counter('api_gateway_rate_limit_exceeded_total',
                                         'Rate limit exceeded',
                                         ['client_id', 'endpoint'])
        self.circuit_breaker_trips = Counter('api_gateway_circuit_breaker_trips_total',
                                           'Circuit breaker trips',
                                           ['endpoint'])
        self.active_connections = Gauge('api_gateway_active_connections',
                                      'Active connections')
        
        # Request tracking
        self.active_requests: Set[str] = set()
        self.request_cache: Dict[str, Any] = {}
        
        logger.info("Enterprise API Gateway initialized")

    async def initialize(self):
        """Initialize the API Gateway components"""
        try:
            # Initialize Redis for caching and rate limiting
            self.redis_client = aioredis.from_url(
                "redis://localhost:6379/7",
                decode_responses=True
            )
            
            # Initialize HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Load default endpoints
            await self._load_default_endpoints()
            
            # Initialize circuit breakers
            await self._initialize_circuit_breakers()
            
            logger.info("API Gateway initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize API Gateway: {e}")
            raise

    async def _load_default_endpoints(self):
        """Load default API endpoints configuration"""
        default_endpoints = [
            APIEndpoint(
                id="user_management",
                name="User Management API",
                path="/api/v1/users",
                method="*",
                protocol=ProtocolType.REST,
                upstream_url="http://user-service:8000",
                authentication=AuthenticationMethod.JWT_TOKEN,
                rate_limit=1000,
                timeout_seconds=30,
                retry_attempts=3,
                circuit_breaker_enabled=True,
                transformation_rules=[],
                caching_enabled=True,
                cache_ttl_seconds=300
            ),
            APIEndpoint(
                id="order_processing",
                name="Order Processing API",
                path="/api/v1/orders",
                method="*",
                protocol=ProtocolType.REST,
                upstream_url="http://order-service:8001",
                authentication=AuthenticationMethod.API_KEY,
                rate_limit=500,
                timeout_seconds=45,
                retry_attempts=5,
                circuit_breaker_enabled=True,
                transformation_rules=[],
                caching_enabled=False
            ),
            APIEndpoint(
                id="legacy_soap_service",
                name="Legacy SOAP Service",
                path="/api/v1/legacy",
                method="POST",
                protocol=ProtocolType.SOAP,
                upstream_url="http://legacy-service:8080/soap",
                authentication=AuthenticationMethod.BASIC_AUTH,
                rate_limit=100,
                timeout_seconds=60,
                retry_attempts=2,
                circuit_breaker_enabled=True,
                transformation_rules=[
                    {
                        "type": "json_to_soap",
                        "envelope_template": "legacy_soap_envelope.xml"
                    }
                ]
            )
        ]
        
        for endpoint in default_endpoints:
            self.endpoints[endpoint.id] = endpoint
            
        logger.info(f"Loaded {len(default_endpoints)} default endpoints")

    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all endpoints"""
        for endpoint_id, endpoint in self.endpoints.items():
            if endpoint.circuit_breaker_enabled:
                self.circuit_breakers[endpoint_id] = CircuitBreakerState(
                    endpoint_id=endpoint_id,
                    state="CLOSED",
                    failure_count=0,
                    last_failure_time=None,
                    next_attempt_time=None,
                    success_threshold=5,
                    failure_threshold=10,
                    timeout_seconds=60
                )

    async def register_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Register a new API endpoint"""
        try:
            self.endpoints[endpoint.id] = endpoint
            
            # Initialize circuit breaker if enabled
            if endpoint.circuit_breaker_enabled:
                self.circuit_breakers[endpoint.id] = CircuitBreakerState(
                    endpoint_id=endpoint.id,
                    state="CLOSED",
                    failure_count=0,
                    last_failure_time=None,
                    next_attempt_time=None,
                    success_threshold=5,
                    failure_threshold=10,
                    timeout_seconds=60
                )
            
            logger.info(f"Registered endpoint: {endpoint.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register endpoint {endpoint.id}: {e}")
            return False

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming API request"""
        request_id = request_data.get('request_id', f"req_{int(time.time() * 1000)}")
        endpoint_id = request_data.get('endpoint_id')
        client_id = request_data.get('client_id', 'anonymous')
        
        try:
            self.active_requests.add(request_id)
            self.active_connections.inc()
            
            # Find matching endpoint
            endpoint = self.endpoints.get(endpoint_id)
            if not endpoint:
                return {
                    "status": "error",
                    "error": "Endpoint not found",
                    "request_id": request_id
                }
            
            # Authenticate request
            auth_result = await self._authenticate_request(request_data, endpoint)
            if not auth_result['authenticated']:
                self.request_counter.labels(
                    endpoint=endpoint_id, 
                    method=request_data.get('method', 'UNKNOWN'),
                    status='401'
                ).inc()
                return {
                    "status": "error",
                    "error": "Authentication failed",
                    "request_id": request_id
                }
            
            # Check rate limits
            rate_limit_result = await self._check_rate_limits(client_id, endpoint_id)
            if not rate_limit_result['allowed']:
                self.rate_limit_exceeded.labels(
                    client_id=client_id,
                    endpoint=endpoint_id
                ).inc()
                return {
                    "status": "error",
                    "error": "Rate limit exceeded",
                    "request_id": request_id,
                    "retry_after": rate_limit_result.get('retry_after', 60)
                }
            
            # Check circuit breaker
            circuit_state = await self._check_circuit_breaker(endpoint_id)
            if circuit_state == "OPEN":
                return {
                    "status": "error",
                    "error": "Service temporarily unavailable",
                    "request_id": request_id
                }
            
            # Check cache if enabled
            if endpoint.caching_enabled:
                cached_response = await self._get_cached_response(request_data, endpoint)
                if cached_response:
                    return cached_response
            
            # Transform request if needed
            transformed_request = await self._transform_request(request_data, endpoint)
            
            # Forward request to upstream service
            start_time = time.time()
            response = await self._forward_request(transformed_request, endpoint)
            duration = time.time() - start_time
            
            # Record metrics
            self.request_duration.labels(
                endpoint=endpoint_id,
                method=request_data.get('method', 'UNKNOWN')
            ).observe(duration)
            
            self.request_counter.labels(
                endpoint=endpoint_id,
                method=request_data.get('method', 'UNKNOWN'),
                status=str(response.get('status_code', 500))
            ).inc()
            
            # Update circuit breaker on success
            if response.get('status_code', 500) < 500:
                await self._record_circuit_breaker_success(endpoint_id)
            else:
                await self._record_circuit_breaker_failure(endpoint_id)
            
            # Transform response
            transformed_response = await self._transform_response(response, endpoint)
            
            # Cache response if enabled and successful
            if endpoint.caching_enabled and response.get('status_code', 500) < 400:
                await self._cache_response(request_data, transformed_response, endpoint)
            
            return transformed_response
            
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            await self._record_circuit_breaker_failure(endpoint_id)
            
            return {
                "status": "error",
                "error": "Internal server error",
                "request_id": request_id
            }
            
        finally:
            self.active_requests.discard(request_id)
            self.active_connections.dec()

    async def _authenticate_request(self, request_data: Dict[str, Any], 
                                  endpoint: APIEndpoint) -> Dict[str, Any]:
        """Authenticate incoming request"""
        try:
            auth_method = endpoint.authentication
            
            if auth_method == AuthenticationMethod.API_KEY:
                api_key = request_data.get('headers', {}).get('X-API-Key')
                if not api_key:
                    return {"authenticated": False, "reason": "Missing API key"}
                
                # Validate API key (simplified)
                is_valid = await self._validate_api_key(api_key)
                return {"authenticated": is_valid, "client_id": api_key[:8]}
                
            elif auth_method == AuthenticationMethod.JWT_TOKEN:
                auth_header = request_data.get('headers', {}).get('Authorization', '')
                if not auth_header.startswith('Bearer '):
                    return {"authenticated": False, "reason": "Missing JWT token"}
                
                token = auth_header[7:]  # Remove 'Bearer ' prefix
                is_valid, payload = await self._validate_jwt_token(token)
                return {
                    "authenticated": is_valid, 
                    "client_id": payload.get('sub', 'unknown') if is_valid else None
                }
                
            elif auth_method == AuthenticationMethod.OAUTH2:
                # OAuth2 validation logic
                access_token = request_data.get('headers', {}).get('Authorization', '').replace('Bearer ', '')
                is_valid = await self._validate_oauth2_token(access_token)
                return {"authenticated": is_valid}
                
            elif auth_method == AuthenticationMethod.BASIC_AUTH:
                auth_header = request_data.get('headers', {}).get('Authorization', '')
                if not auth_header.startswith('Basic '):
                    return {"authenticated": False, "reason": "Missing basic auth"}
                
                # Basic auth validation logic
                credentials = auth_header[6:]  # Remove 'Basic ' prefix
                is_valid = await self._validate_basic_auth(credentials)
                return {"authenticated": is_valid}
            
            # Default allow for testing
            return {"authenticated": True, "client_id": "default"}
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {"authenticated": False, "reason": "Authentication error"}

    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        try:
            # Check in Redis cache first
            cached_result = await self.redis_client.get(f"api_key:{api_key}")
            if cached_result:
                return cached_result == "valid"
            
            # Validate against database or external service
            # For demo purposes, accept keys starting with 'valid_'
            is_valid = api_key.startswith('valid_')
            
            # Cache result for 5 minutes
            await self.redis_client.setex(
                f"api_key:{api_key}", 
                300, 
                "valid" if is_valid else "invalid"
            )
            
            return is_valid
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return False

    async def _validate_jwt_token(self, token: str) -> tuple[bool, Dict[str, Any]]:
        """Validate JWT token"""
        try:
            # For demo purposes, use a simple secret
            secret = "enterprise_gateway_secret"
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            
            # Check expiration
            if payload.get('exp', 0) < time.time():
                return False, {}
                
            return True, payload
            
        except jwt.InvalidTokenError:
            return False, {}
        except Exception as e:
            logger.error(f"JWT validation error: {e}")
            return False, {}

    async def _validate_oauth2_token(self, token: str) -> bool:
        """Validate OAuth2 access token"""
        try:
            # Validate against OAuth2 provider
            # For demo purposes, return True for non-empty tokens
            return bool(token and len(token) > 10)
            
        except Exception as e:
            logger.error(f"OAuth2 validation error: {e}")
            return False

    async def _validate_basic_auth(self, credentials: str) -> bool:
        """Validate basic authentication credentials"""
        try:
            import base64
            decoded = base64.b64decode(credentials).decode('utf-8')
            username, password = decoded.split(':', 1)
            
            # Validate credentials (simplified)
            return username == "admin" and password == "secret"
            
        except Exception as e:
            logger.error(f"Basic auth validation error: {e}")
            return False

    async def _check_rate_limits(self, client_id: str, endpoint_id: str) -> Dict[str, Any]:
        """Check rate limits for client and endpoint"""
        try:
            current_time = int(time.time())
            minute_key = f"rate_limit:{client_id}:{endpoint_id}:minute:{current_time // 60}"
            hour_key = f"rate_limit:{client_id}:{endpoint_id}:hour:{current_time // 3600}"
            day_key = f"rate_limit:{client_id}:{endpoint_id}:day:{current_time // 86400}"
            
            # Get current counts
            minute_count = await self.redis_client.get(minute_key) or 0
            hour_count = await self.redis_client.get(hour_key) or 0
            day_count = await self.redis_client.get(day_key) or 0
            
            minute_count = int(minute_count)
            hour_count = int(hour_count)
            day_count = int(day_count)
            
            # Get rate limit rules (use default if not found)
            endpoint = self.endpoints.get(endpoint_id)
            rate_limit = endpoint.rate_limit if endpoint else 100
            
            # Check limits (simplified - using per-minute limit)
            if minute_count >= rate_limit:
                return {
                    "allowed": False,
                    "retry_after": 60 - (current_time % 60)
                }
            
            # Increment counters
            await self.redis_client.incr(minute_key)
            await self.redis_client.expire(minute_key, 60)
            
            await self.redis_client.incr(hour_key)
            await self.redis_client.expire(hour_key, 3600)
            
            await self.redis_client.incr(day_key)
            await self.redis_client.expire(day_key, 86400)
            
            return {"allowed": True}
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return {"allowed": True}  # Allow on error

    async def _check_circuit_breaker(self, endpoint_id: str) -> str:
        """Check circuit breaker state"""
        try:
            breaker = self.circuit_breakers.get(endpoint_id)
            if not breaker:
                return "CLOSED"
            
            current_time = datetime.utcnow()
            
            if breaker.state == "OPEN":
                if breaker.next_attempt_time and current_time >= breaker.next_attempt_time:
                    breaker.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {endpoint_id} moved to HALF_OPEN")
                    return "HALF_OPEN"
                return "OPEN"
            
            return breaker.state
            
        except Exception as e:
            logger.error(f"Circuit breaker check error: {e}")
            return "CLOSED"

    async def _record_circuit_breaker_success(self, endpoint_id: str):
        """Record successful request for circuit breaker"""
        try:
            breaker = self.circuit_breakers.get(endpoint_id)
            if not breaker:
                return
            
            if breaker.state == "HALF_OPEN":
                breaker.failure_count = 0
                breaker.state = "CLOSED"
                logger.info(f"Circuit breaker for {endpoint_id} closed after successful request")
            elif breaker.state == "CLOSED":
                breaker.failure_count = 0
                
        except Exception as e:
            logger.error(f"Circuit breaker success recording error: {e}")

    async def _record_circuit_breaker_failure(self, endpoint_id: str):
        """Record failed request for circuit breaker"""
        try:
            breaker = self.circuit_breakers.get(endpoint_id)
            if not breaker:
                return
            
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.utcnow()
            
            if breaker.failure_count >= breaker.failure_threshold:
                breaker.state = "OPEN"
                breaker.next_attempt_time = datetime.utcnow() + timedelta(
                    seconds=breaker.timeout_seconds
                )
                
                self.circuit_breaker_trips.labels(endpoint=endpoint_id).inc()
                logger.warning(f"Circuit breaker for {endpoint_id} opened after {breaker.failure_count} failures")
                
        except Exception as e:
            logger.error(f"Circuit breaker failure recording error: {e}")

    async def _get_cached_response(self, request_data: Dict[str, Any], 
                                 endpoint: APIEndpoint) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        try:
            cache_key = self._generate_cache_key(request_data, endpoint)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
                
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    async def _cache_response(self, request_data: Dict[str, Any], 
                            response: Dict[str, Any], endpoint: APIEndpoint):
        """Cache successful response"""
        try:
            cache_key = self._generate_cache_key(request_data, endpoint)
            cached_data = json.dumps(response)
            
            await self.redis_client.setex(
                cache_key,
                endpoint.cache_ttl_seconds,
                cached_data
            )
            
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    def _generate_cache_key(self, request_data: Dict[str, Any], 
                          endpoint: APIEndpoint) -> str:
        """Generate cache key for request"""
        key_data = {
            'endpoint_id': endpoint.id,
            'method': request_data.get('method'),
            'path': request_data.get('path'),
            'query_params': request_data.get('query_params', {}),
            'body': request_data.get('body') if request_data.get('method') == 'GET' else None
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def _transform_request(self, request_data: Dict[str, Any], 
                               endpoint: APIEndpoint) -> Dict[str, Any]:
        """Transform request based on endpoint rules"""
        try:
            transformed_data = request_data.copy()
            
            for rule in endpoint.transformation_rules:
                rule_type = rule.get('type')
                
                if rule_type == 'json_to_soap':
                    transformed_data = await self._transform_json_to_soap(
                        transformed_data, rule
                    )
                elif rule_type == 'field_mapping':
                    transformed_data = await self._apply_field_mapping(
                        transformed_data, rule
                    )
                elif rule_type == 'data_enrichment':
                    transformed_data = await self._enrich_request_data(
                        transformed_data, rule
                    )
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Request transformation error: {e}")
            return request_data

    async def _transform_json_to_soap(self, request_data: Dict[str, Any], 
                                    rule: Dict[str, Any]) -> Dict[str, Any]:
        """Transform JSON request to SOAP envelope"""
        try:
            body = request_data.get('body', {})
            
            # Create SOAP envelope
            envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
    <soap:Header/>
    <soap:Body>
        <Request>
            {self._dict_to_xml(body)}
        </Request>
    </soap:Body>
</soap:Envelope>"""
            
            request_data['body'] = envelope
            request_data['headers'] = request_data.get('headers', {})
            request_data['headers']['Content-Type'] = 'text/xml; charset=utf-8'
            request_data['headers']['SOAPAction'] = rule.get('soap_action', '')
            
            return request_data
            
        except Exception as e:
            logger.error(f"JSON to SOAP transformation error: {e}")
            return request_data

    def _dict_to_xml(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to XML string"""
        xml_parts = []
        for key, value in data.items():
            if isinstance(value, dict):
                xml_parts.append(f"<{key}>{self._dict_to_xml(value)}</{key}>")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        xml_parts.append(f"<{key}>{self._dict_to_xml(item)}</{key}>")
                    else:
                        xml_parts.append(f"<{key}>{item}</{key}>")
            else:
                xml_parts.append(f"<{key}>{value}</{key}>")
        
        return ''.join(xml_parts)

    async def _apply_field_mapping(self, request_data: Dict[str, Any], 
                                 rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field mapping transformation"""
        try:
            mappings = rule.get('mappings', {})
            body = request_data.get('body', {})
            
            if isinstance(body, dict):
                transformed_body = {}
                for source_field, target_field in mappings.items():
                    if source_field in body:
                        transformed_body[target_field] = body[source_field]
                
                # Keep unmapped fields
                for key, value in body.items():
                    if key not in mappings and key not in transformed_body:
                        transformed_body[key] = value
                
                request_data['body'] = transformed_body
            
            return request_data
            
        except Exception as e:
            logger.error(f"Field mapping error: {e}")
            return request_data

    async def _enrich_request_data(self, request_data: Dict[str, Any], 
                                 rule: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich request with additional data"""
        try:
            enrichment_sources = rule.get('sources', [])
            body = request_data.get('body', {})
            
            for source in enrichment_sources:
                if source == 'timestamp':
                    body['timestamp'] = datetime.utcnow().isoformat()
                elif source == 'request_id':
                    body['request_id'] = request_data.get('request_id')
                elif source == 'client_info':
                    body['client_ip'] = request_data.get('client_ip')
                    body['user_agent'] = request_data.get('headers', {}).get('User-Agent')
            
            request_data['body'] = body
            return request_data
            
        except Exception as e:
            logger.error(f"Data enrichment error: {e}")
            return request_data

    async def _forward_request(self, request_data: Dict[str, Any], 
                             endpoint: APIEndpoint) -> Dict[str, Any]:
        """Forward request to upstream service"""
        try:
            url = f"{endpoint.upstream_url.rstrip('/')}{request_data.get('path', '')}"
            method = request_data.get('method', 'GET')
            headers = request_data.get('headers', {})
            body = request_data.get('body')
            params = request_data.get('query_params', {})
            
            # Handle different protocols
            if endpoint.protocol == ProtocolType.SOAP:
                return await self._forward_soap_request(url, headers, body, endpoint)
            elif endpoint.protocol == ProtocolType.GRPC:
                return await self._forward_grpc_request(url, headers, body, endpoint)
            else:
                # REST/GraphQL
                return await self._forward_http_request(
                    method, url, headers, body, params, endpoint
                )
                
        except Exception as e:
            logger.error(f"Request forwarding error: {e}")
            return {
                "status_code": 500,
                "body": {"error": "Request forwarding failed"},
                "headers": {}
            }

    async def _forward_http_request(self, method: str, url: str, headers: Dict[str, str],
                                  body: Any, params: Dict[str, str], 
                                  endpoint: APIEndpoint) -> Dict[str, Any]:
        """Forward HTTP request"""
        try:
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, dict) else None,
                    data=body if isinstance(body, str) else None,
                    params=params
                ) as response:
                    
                    response_body = await response.text()
                    response_headers = dict(response.headers)
                    
                    # Try to parse as JSON
                    try:
                        response_body = json.loads(response_body)
                    except json.JSONDecodeError:
                        pass  # Keep as string
                    
                    return {
                        "status_code": response.status,
                        "body": response_body,
                        "headers": response_headers
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {url}")
            return {
                "status_code": 504,
                "body": {"error": "Gateway timeout"},
                "headers": {}
            }
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return {
                "status_code": 500,
                "body": {"error": "Internal server error"},
                "headers": {}
            }

    async def _forward_soap_request(self, url: str, headers: Dict[str, str],
                                  body: str, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Forward SOAP request"""
        try:
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=url,
                    headers=headers,
                    data=body
                ) as response:
                    
                    response_body = await response.text()
                    response_headers = dict(response.headers)
                    
                    return {
                        "status_code": response.status,
                        "body": response_body,
                        "headers": response_headers
                    }
                    
        except Exception as e:
            logger.error(f"SOAP request error: {e}")
            return {
                "status_code": 500,
                "body": {"error": "SOAP request failed"},
                "headers": {}
            }

    async def _forward_grpc_request(self, url: str, headers: Dict[str, str],
                                  body: Any, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Forward gRPC request (simplified)"""
        try:
            # gRPC forwarding would require proper gRPC client implementation
            # For now, simulate with HTTP/2
            return {
                "status_code": 200,
                "body": {"message": "gRPC forwarding not fully implemented"},
                "headers": {}
            }
            
        except Exception as e:
            logger.error(f"gRPC request error: {e}")
            return {
                "status_code": 500,
                "body": {"error": "gRPC request failed"},
                "headers": {}
            }

    async def _transform_response(self, response: Dict[str, Any], 
                                endpoint: APIEndpoint) -> Dict[str, Any]:
        """Transform response based on endpoint rules"""
        try:
            transformed_response = response.copy()
            
            for rule in endpoint.transformation_rules:
                rule_type = rule.get('type')
                
                if rule_type == 'soap_to_json':
                    transformed_response = await self._transform_soap_to_json(
                        transformed_response, rule
                    )
                elif rule_type == 'response_mapping':
                    transformed_response = await self._apply_response_mapping(
                        transformed_response, rule
                    )
            
            return transformed_response
            
        except Exception as e:
            logger.error(f"Response transformation error: {e}")
            return response

    async def _transform_soap_to_json(self, response: Dict[str, Any], 
                                    rule: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SOAP response to JSON"""
        try:
            body = response.get('body', '')
            if isinstance(body, str) and body.strip().startswith('<?xml'):
                # Parse XML and convert to JSON
                root = ET.fromstring(body)
                json_data = self._xml_to_dict(root)
                response['body'] = json_data
                
                # Update content type
                headers = response.get('headers', {})
                headers['Content-Type'] = 'application/json'
                response['headers'] = headers
            
            return response
            
        except Exception as e:
            logger.error(f"SOAP to JSON transformation error: {e}")
            return response

    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Handle attributes
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # Handle text content
        if element.text and element.text.strip():
            if len(element) == 0:  # Leaf node
                return element.text.strip()
            else:
                result['#text'] = element.text.strip()
        
        # Handle child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                # Convert to list if multiple elements with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result if result else None

    async def _apply_response_mapping(self, response: Dict[str, Any], 
                                    rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply response field mapping"""
        try:
            mappings = rule.get('mappings', {})
            body = response.get('body', {})
            
            if isinstance(body, dict):
                transformed_body = {}
                for source_field, target_field in mappings.items():
                    if source_field in body:
                        transformed_body[target_field] = body[source_field]
                
                response['body'] = transformed_body
            
            return response
            
        except Exception as e:
            logger.error(f"Response mapping error: {e}")
            return response

    async def get_endpoint_health(self) -> Dict[str, Any]:
        """Get health status of all endpoints"""
        try:
            health_status = {}
            
            for endpoint_id, endpoint in self.endpoints.items():
                circuit_breaker = self.circuit_breakers.get(endpoint_id)
                
                health_status[endpoint_id] = {
                    "name": endpoint.name,
                    "protocol": endpoint.protocol.value,
                    "upstream_url": endpoint.upstream_url,
                    "circuit_breaker_state": circuit_breaker.state if circuit_breaker else "N/A",
                    "failure_count": circuit_breaker.failure_count if circuit_breaker else 0,
                    "last_failure": circuit_breaker.last_failure_time.isoformat() if circuit_breaker and circuit_breaker.last_failure_time else None
                }
            
            return {
                "status": "healthy",
                "endpoints": health_status,
                "active_requests": len(self.active_requests),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            return {
                "active_connections": len(self.active_requests),
                "total_endpoints": len(self.endpoints),
                "circuit_breakers": {
                    endpoint_id: breaker.state 
                    for endpoint_id, breaker in self.circuit_breakers.items()
                },
                "cache_stats": await self._get_cache_statistics(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Metrics summary error: {e}")
            return {"error": str(e)}

    async def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        try:
            # Get cache keys count
            cache_keys = await self.redis_client.keys("cache:*")
            
            return {
                "cached_responses": len(cache_keys),
                "cache_hit_rate": "Not implemented",  # Would require tracking
                "total_cache_size": "Not implemented"
            }
            
        except Exception as e:
            logger.error(f"Cache statistics error: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session:
                await self.session.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("API Gateway cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Example usage and testing
async def main():
    gateway = EnterpriseAPIGateway()
    await gateway.initialize()
    
    # Test request
    test_request = {
        "request_id": "test_001",
        "endpoint_id": "user_management",
        "method": "GET",
        "path": "/api/v1/users/123",
        "headers": {
            "Authorization": "Bearer valid_jwt_token",
            "Content-Type": "application/json"
        },
        "client_id": "test_client",
        "client_ip": "127.0.0.1"
    }
    
    response = await gateway.process_request(test_request)
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Get health status
    health = await gateway.get_endpoint_health()
    print(f"Health: {json.dumps(health, indent=2)}")
    
    await gateway.cleanup()

if __name__ == "__main__":
    asyncio.run(main())