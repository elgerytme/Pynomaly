"""
Developer SDK and API Documentation for Pynomaly Detection
==========================================================

Comprehensive developer tools providing:
- Python SDK for easy integration
- REST API client with authentication
- Code generation utilities
- Interactive API documentation
- SDK examples and tutorials
- Testing and debugging tools
- Performance monitoring utilities
"""

import logging
import json
import time
import inspect
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

logger = logging.getLogger(__name__)

class SDKLanguage(Enum):
    """SDK language enumeration."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"

class APIMethod(Enum):
    """API method enumeration."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

class ResponseFormat(Enum):
    """Response format enumeration."""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    BINARY = "binary"

@dataclass
class APIEndpoint:
    """API endpoint definition."""
    endpoint_id: str
    path: str
    method: APIMethod
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_schema: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    authentication_required: bool = True
    rate_limit: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    deprecated: bool = False
    version: str = "v1"

@dataclass
class SDKMethod:
    """SDK method definition."""
    method_id: str
    name: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = "dict"
    example_code: str = ""
    related_endpoints: List[str] = field(default_factory=list)
    category: str = "general"
    complexity: str = "beginner"  # beginner, intermediate, advanced

@dataclass
class CodeExample:
    """Code example definition."""
    example_id: str
    title: str
    description: str
    language: SDKLanguage
    code: str
    expected_output: str = ""
    requirements: List[str] = field(default_factory=list)
    category: str = "general"
    complexity: str = "beginner"

class DeveloperSDK:
    """Comprehensive developer SDK for Pynomaly Detection."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize SDK.
        
        Args:
            base_url: API base URL
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
        # HTTP client configuration
        self.session = None
        self.timeout = 30
        self.retry_attempts = 3
        
        # SDK state
        self.authenticated = False
        self.user_info = {}
        
        # Performance tracking
        self.request_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'last_request_time': None
        }
        
        # Initialize HTTP session
        self._initialize_session()
        
        logger.info(f"Developer SDK initialized (base_url: {base_url})")
    
    def authenticate(self, api_key: Optional[str] = None, 
                    username: Optional[str] = None, 
                    password: Optional[str] = None) -> bool:
        """Authenticate with the API.
        
        Args:
            api_key: API key for authentication
            username: Username for basic auth
            password: Password for basic auth
            
        Returns:
            True if authentication successful
        """
        try:
            if api_key:
                self.api_key = api_key
                self.session.headers.update({'Authorization': f'Bearer {api_key}'})
            elif username and password:
                # Basic authentication
                response = self._request('POST', '/auth/login', {
                    'username': username,
                    'password': password
                })
                
                if response and response.get('token'):
                    self.api_key = response['token']
                    self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
            
            # Verify authentication
            user_info = self.get_user_info()
            if user_info:
                self.authenticated = True
                self.user_info = user_info
                logger.info("SDK authentication successful")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"SDK authentication failed: {e}")
            return False
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get authenticated user information.
        
        Returns:
            User information dictionary or None
        """
        return self._request('GET', '/user/profile')
    
    # Anomaly Detection Methods
    def detect_anomalies(self, data: Union[List, Dict], 
                        algorithm: str = 'auto',
                        config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Detect anomalies in data.
        
        Args:
            data: Input data for anomaly detection
            algorithm: Detection algorithm to use
            config: Optional algorithm configuration
            
        Returns:
            Detection results or None
        """
        payload = {
            'data': data,
            'algorithm': algorithm,
            'config': config or {}
        }
        
        return self._request('POST', '/detection/analyze', payload)
    
    def batch_detect(self, datasets: List[Dict[str, Any]], 
                    config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Perform batch anomaly detection.
        
        Args:
            datasets: List of datasets to process
            config: Optional batch configuration
            
        Returns:
            Batch processing results or None
        """
        payload = {
            'datasets': datasets,
            'config': config or {}
        }
        
        return self._request('POST', '/detection/batch', payload)
    
    def get_algorithms(self) -> Optional[List[Dict[str, Any]]]:
        """Get available detection algorithms.
        
        Returns:
            List of available algorithms or None
        """
        return self._request('GET', '/algorithms')
    
    def get_algorithm_info(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed algorithm information.
        
        Args:
            algorithm_id: Algorithm identifier
            
        Returns:
            Algorithm information or None
        """
        return self._request('GET', f'/algorithms/{algorithm_id}')
    
    # Model Management Methods
    def create_model(self, model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new anomaly detection model.
        
        Args:
            model_data: Model configuration and training data
            
        Returns:
            Model creation result or None
        """
        return self._request('POST', '/models', model_data)
    
    def get_models(self, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """Get user's models.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of models or None
        """
        params = {'limit': limit}
        return self._request('GET', '/models', params=params)
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get specific model details.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model details or None
        """
        return self._request('GET', f'/models/{model_id}')
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update model configuration.
        
        Args:
            model_id: Model identifier
            updates: Model updates
            
        Returns:
            Updated model information or None
        """
        return self._request('PUT', f'/models/{model_id}', updates)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deletion successful
        """
        response = self._request('DELETE', f'/models/{model_id}')
        return response is not None
    
    def predict(self, model_id: str, data: Union[List, Dict]) -> Optional[Dict[str, Any]]:
        """Make predictions using trained model.
        
        Args:
            model_id: Model identifier
            data: Input data for prediction
            
        Returns:
            Prediction results or None
        """
        payload = {'data': data}
        return self._request('POST', f'/models/{model_id}/predict', payload)
    
    # Data Management Methods
    def upload_dataset(self, dataset_name: str, data: Union[List, Dict],
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Upload dataset for processing.
        
        Args:
            dataset_name: Name for the dataset
            data: Dataset content
            metadata: Optional dataset metadata
            
        Returns:
            Upload result or None
        """
        payload = {
            'name': dataset_name,
            'data': data,
            'metadata': metadata or {}
        }
        
        return self._request('POST', '/datasets', payload)
    
    def get_datasets(self, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """Get user's datasets.
        
        Args:
            limit: Maximum number of datasets to return
            
        Returns:
            List of datasets or None
        """
        params = {'limit': limit}
        return self._request('GET', '/datasets', params=params)
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get specific dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dataset information or None
        """
        return self._request('GET', f'/datasets/{dataset_id}')
    
    # Project Management Methods
    def create_project(self, project_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new project.
        
        Args:
            project_data: Project information
            
        Returns:
            Project creation result or None
        """
        return self._request('POST', '/projects', project_data)
    
    def get_projects(self, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """Get user's projects.
        
        Args:
            limit: Maximum number of projects to return
            
        Returns:
            List of projects or None
        """
        params = {'limit': limit}
        return self._request('GET', '/projects', params=params)
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get specific project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project details or None
        """
        return self._request('GET', f'/projects/{project_id}')
    
    # Monitoring and Analytics Methods
    def get_usage_statistics(self) -> Optional[Dict[str, Any]]:
        """Get API usage statistics.
        
        Returns:
            Usage statistics or None
        """
        return self._request('GET', '/analytics/usage')
    
    def get_performance_metrics(self, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get performance metrics.
        
        Args:
            model_id: Optional specific model to get metrics for
            
        Returns:
            Performance metrics or None
        """
        params = {}
        if model_id:
            params['model_id'] = model_id
        
        return self._request('GET', '/analytics/performance', params=params)
    
    # Utility Methods
    def health_check(self) -> bool:
        """Check API health status.
        
        Returns:
            True if API is healthy
        """
        try:
            response = self._request('GET', '/health')
            return response is not None and response.get('status') == 'healthy'
        except Exception:
            return False
    
    def get_api_info(self) -> Optional[Dict[str, Any]]:
        """Get API information.
        
        Returns:
            API information or None
        """
        return self._request('GET', '/info')
    
    def validate_data(self, data: Union[List, Dict],
                     schema_type: str = 'detection') -> Optional[Dict[str, Any]]:
        """Validate data format.
        
        Args:
            data: Data to validate
            schema_type: Schema type to validate against
            
        Returns:
            Validation result or None
        """
        payload = {
            'data': data,
            'schema_type': schema_type
        }
        
        return self._request('POST', '/validation/data', payload)
    
    def get_sdk_metrics(self) -> Dict[str, Any]:
        """Get SDK performance metrics.
        
        Returns:
            SDK metrics dictionary
        """
        return self.request_metrics.copy()
    
    def _initialize_session(self):
        """Initialize HTTP session."""
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'User-Agent': 'Pynomaly-SDK/1.0.0'
            })
            
            if self.api_key:
                self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
    
    def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None,
                params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Make HTTP request to API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            params: URL parameters
            
        Returns:
            Response data or None
        """
        if not REQUESTS_AVAILABLE or not self.session:
            logger.error("HTTP client not available")
            return None
        
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            # Update metrics
            self.request_metrics['total_requests'] += 1
            
            # Make request
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, params=params, timeout=self.timeout)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data, params=params, timeout=self.timeout)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=params, timeout=self.timeout)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            # Update response time
            response_time = time.time() - start_time
            self.request_metrics['average_response_time'] = (
                self.request_metrics['average_response_time'] * 0.9 + response_time * 0.1
            )
            self.request_metrics['last_request_time'] = datetime.now()
            
            # Handle response
            if response.status_code >= 200 and response.status_code < 300:
                self.request_metrics['successful_requests'] += 1
                
                if response.content:
                    return response.json()
                else:
                    return {}
            else:
                self.request_metrics['failed_requests'] += 1
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.request_metrics['failed_requests'] += 1
            logger.error(f"HTTP request failed: {e}")
            return None


class APIDocumentation:
    """Interactive API documentation generator and manager."""
    
    def __init__(self):
        """Initialize API documentation system."""
        # API specification
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.sdk_methods: Dict[str, SDKMethod] = {}
        self.code_examples: Dict[str, CodeExample] = {}
        
        # Documentation templates
        self.templates = {}
        
        # Generated documentation
        self.generated_docs: Dict[str, str] = {}
        
        # Initialize API specification
        self._initialize_api_spec()
        
        logger.info("API Documentation initialized")
    
    def add_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Add API endpoint to documentation.
        
        Args:
            endpoint: API endpoint definition
            
        Returns:
            True if added successfully
        """
        try:
            self.endpoints[endpoint.endpoint_id] = endpoint
            logger.info(f"API endpoint added: {endpoint.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add API endpoint: {e}")
            return False
    
    def add_sdk_method(self, method: SDKMethod) -> bool:
        """Add SDK method to documentation.
        
        Args:
            method: SDK method definition
            
        Returns:
            True if added successfully
        """
        try:
            self.sdk_methods[method.method_id] = method
            logger.info(f"SDK method added: {method.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add SDK method: {e}")
            return False
    
    def add_code_example(self, example: CodeExample) -> bool:
        """Add code example to documentation.
        
        Args:
            example: Code example definition
            
        Returns:
            True if added successfully
        """
        try:
            self.code_examples[example.example_id] = example
            logger.info(f"Code example added: {example.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add code example: {e}")
            return False
    
    def generate_api_reference(self, format: str = 'markdown') -> str:
        """Generate API reference documentation.
        
        Args:
            format: Documentation format (markdown, html, json)
            
        Returns:
            Generated documentation
        """
        try:
            if format == 'markdown':
                return self._generate_markdown_reference()
            elif format == 'html':
                return self._generate_html_reference()
            elif format == 'json':
                return self._generate_json_reference()
            else:
                logger.error(f"Unsupported format: {format}")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to generate API reference: {e}")
            return ""
    
    def generate_sdk_guide(self, language: SDKLanguage = SDKLanguage.PYTHON) -> str:
        """Generate SDK usage guide.
        
        Args:
            language: Programming language for the guide
            
        Returns:
            Generated SDK guide
        """
        try:
            return self._generate_sdk_guide(language)
            
        except Exception as e:
            logger.error(f"Failed to generate SDK guide: {e}")
            return ""
    
    def generate_code_samples(self, category: Optional[str] = None,
                            language: Optional[SDKLanguage] = None) -> Dict[str, str]:
        """Generate code samples.
        
        Args:
            category: Optional category filter
            language: Optional language filter
            
        Returns:
            Dictionary of code samples
        """
        try:
            samples = {}
            
            for example_id, example in self.code_examples.items():
                # Apply filters
                if category and example.category != category:
                    continue
                
                if language and example.language != language:
                    continue
                
                samples[example_id] = {
                    'title': example.title,
                    'description': example.description,
                    'language': example.language.value,
                    'code': example.code,
                    'expected_output': example.expected_output,
                    'complexity': example.complexity
                }
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to generate code samples: {e}")
            return {}
    
    def create_interactive_docs(self) -> Dict[str, Any]:
        """Create interactive API documentation data.
        
        Returns:
            Interactive documentation structure
        """
        try:
            docs = {
                'info': {
                    'title': 'Pynomaly Detection API',
                    'version': '1.0.0',
                    'description': 'Comprehensive anomaly detection platform',
                    'contact': {
                        'name': 'Pynomaly Team',
                        'email': 'support@pynomaly.com',
                        'url': 'https://pynomaly.com'
                    }
                },
                'servers': [
                    {
                        'url': 'https://api.pynomaly.com/v1',
                        'description': 'Production server'
                    },
                    {
                        'url': 'https://staging-api.pynomaly.com/v1',
                        'description': 'Staging server'
                    }
                ],
                'paths': {},
                'components': {
                    'securitySchemes': {
                        'BearerAuth': {
                            'type': 'http',
                            'scheme': 'bearer',
                            'bearerFormat': 'JWT'
                        }
                    }
                },
                'security': [{'BearerAuth': []}]
            }
            
            # Add endpoints
            for endpoint in self.endpoints.values():
                path = endpoint.path
                
                if path not in docs['paths']:
                    docs['paths'][path] = {}
                
                method = endpoint.method.value.lower()
                
                docs['paths'][path][method] = {
                    'summary': endpoint.description,
                    'operationId': endpoint.endpoint_id,
                    'parameters': self._format_parameters(endpoint.parameters),
                    'requestBody': self._format_request_body(endpoint.request_schema) if endpoint.request_schema else None,
                    'responses': self._format_responses(endpoint.response_schema),
                    'security': [{'BearerAuth': []}] if endpoint.authentication_required else [],
                    'examples': endpoint.examples
                }
                
                if endpoint.deprecated:
                    docs['paths'][path][method]['deprecated'] = True
            
            return docs
            
        except Exception as e:
            logger.error(f"Failed to create interactive docs: {e}")
            return {}
    
    def validate_documentation(self) -> Dict[str, List[str]]:
        """Validate documentation completeness.
        
        Returns:
            Validation results with issues found
        """
        issues = {
            'missing_descriptions': [],
            'missing_examples': [],
            'missing_schemas': [],
            'broken_links': []
        }
        
        try:
            # Check endpoints
            for endpoint_id, endpoint in self.endpoints.items():
                if not endpoint.description.strip():
                    issues['missing_descriptions'].append(f"Endpoint {endpoint_id}")
                
                if not endpoint.examples:
                    issues['missing_examples'].append(f"Endpoint {endpoint_id}")
                
                if not endpoint.request_schema and endpoint.method in [APIMethod.POST, APIMethod.PUT]:
                    issues['missing_schemas'].append(f"Endpoint {endpoint_id} request schema")
                
                if not endpoint.response_schema:
                    issues['missing_schemas'].append(f"Endpoint {endpoint_id} response schema")
            
            # Check SDK methods
            for method_id, method in self.sdk_methods.items():
                if not method.description.strip():
                    issues['missing_descriptions'].append(f"SDK method {method_id}")
                
                if not method.example_code.strip():
                    issues['missing_examples'].append(f"SDK method {method_id}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Documentation validation failed: {e}")
            return issues
    
    def _initialize_api_spec(self):
        """Initialize API specification with core endpoints."""
        # Core detection endpoints
        core_endpoints = [
            APIEndpoint(
                endpoint_id="detect_anomalies",
                path="/detection/analyze",
                method=APIMethod.POST,
                description="Detect anomalies in provided data",
                request_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "array"},
                        "algorithm": {"type": "string", "default": "auto"},
                        "config": {"type": "object"}
                    },
                    "required": ["data"]
                },
                response_schema={
                    "type": "object",
                    "properties": {
                        "anomalies": {"type": "array"},
                        "scores": {"type": "array"},
                        "algorithm_used": {"type": "string"},
                        "processing_time": {"type": "number"}
                    }
                },
                examples=[{
                    "request": {
                        "data": [[1, 2], [2, 3], [10, 15], [1, 1]],
                        "algorithm": "isolation_forest"
                    },
                    "response": {
                        "anomalies": [2],
                        "scores": [-0.1, -0.15, -0.8, -0.12],
                        "algorithm_used": "isolation_forest",
                        "processing_time": 0.045
                    }
                }]
            ),
            
            APIEndpoint(
                endpoint_id="get_algorithms",
                path="/algorithms",
                method=APIMethod.GET,
                description="Get list of available detection algorithms",
                response_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "object"}
                        }
                    }
                },
                authentication_required=False
            ),
            
            APIEndpoint(
                endpoint_id="create_model",
                path="/models",
                method=APIMethod.POST,
                description="Create and train a new anomaly detection model",
                request_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "algorithm": {"type": "string"},
                        "training_data": {"type": "array"},
                        "config": {"type": "object"}
                    },
                    "required": ["name", "algorithm", "training_data"]
                },
                response_schema={
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string"},
                        "name": {"type": "string"},
                        "status": {"type": "string"},
                        "created_at": {"type": "string", "format": "date-time"}
                    }
                }
            )
        ]
        
        for endpoint in core_endpoints:
            self.endpoints[endpoint.endpoint_id] = endpoint
        
        # Core SDK methods
        core_methods = [
            SDKMethod(
                method_id="detect_anomalies",
                name="detect_anomalies",
                description="Detect anomalies in data using specified algorithm",
                parameters=[
                    {"name": "data", "type": "Union[List, Dict]", "required": True, "description": "Input data"},
                    {"name": "algorithm", "type": "str", "required": False, "default": "auto", "description": "Detection algorithm"},
                    {"name": "config", "type": "Optional[Dict]", "required": False, "description": "Algorithm configuration"}
                ],
                return_type="Optional[Dict[str, Any]]",
                example_code='''
# Basic anomaly detection
client = DeveloperSDK("https://api.pynomaly.com", "your-api-key")
data = [[1, 2], [2, 3], [10, 15], [1, 1]]
result = client.detect_anomalies(data)
print(f"Anomalies found at indices: {result['anomalies']}")
''',
                related_endpoints=["detect_anomalies"],
                category="detection",
                complexity="beginner"
            ),
            
            SDKMethod(
                method_id="create_model",
                name="create_model",
                description="Create and train a new anomaly detection model",
                parameters=[
                    {"name": "model_data", "type": "Dict[str, Any]", "required": True, "description": "Model configuration and training data"}
                ],
                return_type="Optional[Dict[str, Any]]",
                example_code='''
# Create a custom model
model_data = {
    "name": "My Anomaly Model",
    "algorithm": "isolation_forest",
    "training_data": training_dataset,
    "config": {"contamination": 0.1}
}
result = client.create_model(model_data)
model_id = result["model_id"]
''',
                related_endpoints=["create_model"],
                category="models",
                complexity="intermediate"
            )
        ]
        
        for method in core_methods:
            self.sdk_methods[method.method_id] = method
        
        # Core code examples
        core_examples = [
            CodeExample(
                example_id="basic_detection",
                title="Basic Anomaly Detection",
                description="Simple example of detecting anomalies in a dataset",
                language=SDKLanguage.PYTHON,
                code='''
from pynomaly.sdk import DeveloperSDK

# Initialize SDK
client = DeveloperSDK("https://api.pynomaly.com", "your-api-key")

# Sample data with one outlier
data = [
    [1, 2], [2, 3], [3, 4], [2, 2], [1, 3],  # Normal points
    [10, 15]  # Outlier
]

# Detect anomalies
result = client.detect_anomalies(data, algorithm="isolation_forest")

# Display results
print(f"Anomalies detected at indices: {result['anomalies']}")
print(f"Anomaly scores: {result['scores']}")
''',
                expected_output="Anomalies detected at indices: [5]\\nAnomaly scores: [-0.1, -0.1, -0.1, -0.1, -0.1, -0.8]",
                category="detection",
                complexity="beginner"
            ),
            
            CodeExample(
                example_id="batch_processing",
                title="Batch Processing Multiple Datasets",
                description="Process multiple datasets in a single API call",
                language=SDKLanguage.PYTHON,
                code='''
# Prepare multiple datasets
datasets = [
    {
        "name": "dataset1",
        "data": [[1, 2], [2, 3], [10, 15]],
        "algorithm": "isolation_forest"
    },
    {
        "name": "dataset2", 
        "data": [[5, 6], [6, 7], [20, 25]],
        "algorithm": "one_class_svm"
    }
]

# Process all datasets
results = client.batch_detect(datasets)

# Review results
for dataset_result in results["results"]:
    print(f"Dataset: {dataset_result['name']}")
    print(f"Anomalies: {dataset_result['anomalies']}")
''',
                category="detection",
                complexity="intermediate"
            )
        ]
        
        for example in core_examples:
            self.code_examples[example.example_id] = example
    
    def _generate_markdown_reference(self) -> str:
        """Generate markdown API reference."""
        md_content = """# Pynomaly Detection API Reference

## Overview

The Pynomaly Detection API provides comprehensive anomaly detection capabilities through a RESTful interface.

## Authentication

All API requests require authentication using a Bearer token:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

"""
        
        # Group endpoints by category
        categories = {}
        for endpoint in self.endpoints.values():
            category = endpoint.path.split('/')[1] if '/' in endpoint.path else 'general'
            if category not in categories:
                categories[category] = []
            categories[category].append(endpoint)
        
        for category, endpoints in categories.items():
            md_content += f"\n### {category.title()}\n\n"
            
            for endpoint in endpoints:
                md_content += f"#### {endpoint.method.value} {endpoint.path}\n\n"
                md_content += f"{endpoint.description}\n\n"
                
                if endpoint.parameters:
                    md_content += "**Parameters:**\n\n"
                    for param, details in endpoint.parameters.items():
                        md_content += f"- `{param}` ({details.get('type', 'string')}): {details.get('description', '')}\n"
                    md_content += "\n"
                
                if endpoint.examples:
                    md_content += "**Example:**\n\n"
                    example = endpoint.examples[0]
                    if 'request' in example:
                        md_content += "Request:\n```json\n" + json.dumps(example['request'], indent=2) + "\n```\n\n"
                    if 'response' in example:
                        md_content += "Response:\n```json\n" + json.dumps(example['response'], indent=2) + "\n```\n\n"
        
        return md_content
    
    def _generate_html_reference(self) -> str:
        """Generate HTML API reference."""
        # Simplified HTML generation
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Pynomaly API Reference</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .endpoint { border: 1px solid #ddd; margin: 20px 0; padding: 20px; }
        .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>Pynomaly Detection API Reference</h1>
"""
        
        for endpoint in self.endpoints.values():
            html_content += f'''
    <div class="endpoint">
        <h3><span class="method">{endpoint.method.value}</span> {endpoint.path}</h3>
        <p>{endpoint.description}</p>
        
        {f'<h4>Parameters:</h4><ul>' + ''.join([f'<li><code>{p}</code>: {d.get("description", "")}</li>' for p, d in endpoint.parameters.items()]) + '</ul>' if endpoint.parameters else ''}
        
        {f'<h4>Example:</h4><pre>{json.dumps(endpoint.examples[0], indent=2)}</pre>' if endpoint.examples else ''}
    </div>
'''
        
        html_content += "</body></html>"
        return html_content
    
    def _generate_json_reference(self) -> str:
        """Generate JSON API reference."""
        spec = {
            'endpoints': {},
            'sdk_methods': {},
            'examples': {}
        }
        
        for endpoint_id, endpoint in self.endpoints.items():
            spec['endpoints'][endpoint_id] = {
                'path': endpoint.path,
                'method': endpoint.method.value,
                'description': endpoint.description,
                'parameters': endpoint.parameters,
                'request_schema': endpoint.request_schema,
                'response_schema': endpoint.response_schema,
                'examples': endpoint.examples
            }
        
        for method_id, method in self.sdk_methods.items():
            spec['sdk_methods'][method_id] = {
                'name': method.name,
                'description': method.description,
                'parameters': method.parameters,
                'return_type': method.return_type,
                'example_code': method.example_code
            }
        
        for example_id, example in self.code_examples.items():
            spec['examples'][example_id] = {
                'title': example.title,
                'description': example.description,
                'language': example.language.value,
                'code': example.code,
                'category': example.category
            }
        
        return json.dumps(spec, indent=2)
    
    def _generate_sdk_guide(self, language: SDKLanguage) -> str:
        """Generate SDK usage guide for specific language."""
        if language == SDKLanguage.PYTHON:
            return self._generate_python_guide()
        else:
            return f"# {language.value.title()} SDK Guide\n\nComing soon..."
    
    def _generate_python_guide(self) -> str:
        """Generate Python SDK guide."""
        return '''# Pynomaly Python SDK Guide

## Installation

```bash
pip install pynomaly-sdk
```

## Quick Start

```python
from pynomaly.sdk import DeveloperSDK

# Initialize the SDK
client = DeveloperSDK("https://api.pynomaly.com", "your-api-key")

# Authenticate (if not using API key in constructor)
client.authenticate("your-api-key")

# Basic anomaly detection
data = [[1, 2], [2, 3], [10, 15]]
result = client.detect_anomalies(data)
print(result)
```

## Advanced Usage

### Custom Model Training

```python
# Create and train a custom model
model_data = {
    "name": "Production Model",
    "algorithm": "isolation_forest", 
    "training_data": training_dataset,
    "config": {"contamination": 0.1}
}

model = client.create_model(model_data)
model_id = model["model_id"]

# Use the model for predictions
predictions = client.predict(model_id, test_data)
```

### Batch Processing

```python
# Process multiple datasets
datasets = [
    {"name": "batch1", "data": dataset1},
    {"name": "batch2", "data": dataset2}
]

results = client.batch_detect(datasets)
```

## Error Handling

```python
try:
    result = client.detect_anomalies(data)
    if result is None:
        print("API request failed")
    else:
        print(f"Found {len(result['anomalies'])} anomalies")
except Exception as e:
    print(f"Error: {e}")
```
'''
    
    def _format_parameters(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format parameters for OpenAPI spec."""
        formatted = []
        for name, details in parameters.items():
            formatted.append({
                'name': name,
                'in': details.get('in', 'query'),
                'required': details.get('required', False),
                'schema': {'type': details.get('type', 'string')},
                'description': details.get('description', '')
            })
        return formatted
    
    def _format_request_body(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Format request body for OpenAPI spec."""
        return {
            'required': True,
            'content': {
                'application/json': {
                    'schema': schema
                }
            }
        }
    
    def _format_responses(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Format responses for OpenAPI spec."""
        return {
            '200': {
                'description': 'Successful response',
                'content': {
                    'application/json': {
                        'schema': schema
                    }
                }
            },
            '400': {
                'description': 'Bad request'
            },
            '401': {
                'description': 'Unauthorized'
            },
            '500': {
                'description': 'Internal server error'
            }
        }