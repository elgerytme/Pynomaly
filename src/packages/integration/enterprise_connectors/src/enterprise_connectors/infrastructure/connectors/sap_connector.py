import asyncio
import logging
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aioredis
from prometheus_client import Counter, Histogram, Gauge
import xmltodict
import requests

logger = logging.getLogger(__name__)

class SAPSystemType(Enum):
    ERP = "erp"
    S4_HANA = "s4_hana"
    BW = "bw"
    CRM = "crm"
    SRM = "srm"
    PI_PO = "pi_po"
    FIORI = "fiori"

class AuthenticationType(Enum):
    BASIC = "basic"
    OAUTH2 = "oauth2"
    SAML = "saml"
    X509 = "x509"

class IntegrationPattern(Enum):
    RFC = "rfc"
    BAPI = "bapi"
    IDOC = "idoc"
    REST_API = "rest_api"
    ODATA = "odata"
    SOAP = "soap"

@dataclass
class SAPConnection:
    connection_id: str
    name: str
    host: str
    port: int
    client: str
    system_id: str
    system_type: SAPSystemType
    authentication: AuthenticationType
    username: str
    password: str
    language: str = "EN"
    pool_size: int = 10
    timeout_seconds: int = 30
    ssl_enabled: bool = True
    router_string: Optional[str] = None

@dataclass
class SAPOperation:
    operation_id: str
    name: str
    pattern: IntegrationPattern
    function_module: Optional[str] = None
    bapi_name: Optional[str] = None
    idoc_type: Optional[str] = None
    odata_service: Optional[str] = None
    endpoint_path: Optional[str] = None
    input_mapping: Dict[str, Any] = None
    output_mapping: Dict[str, Any] = None
    retry_attempts: int = 3
    circuit_breaker_enabled: bool = True

@dataclass
class SAPRequest:
    request_id: str
    connection_id: str
    operation_id: str
    parameters: Dict[str, Any]
    headers: Optional[Dict[str, str]] = None
    priority: int = 5
    timeout_override: Optional[int] = None

@dataclass
class SAPResponse:
    request_id: str
    success: bool
    data: Any
    error_message: Optional[str] = None
    execution_time_ms: int = 0
    sap_return_code: Optional[str] = None
    warnings: List[str] = None

class SAPConnector:
    def __init__(self):
        self.connections: Dict[str, SAPConnection] = {}
        self.operations: Dict[str, SAPOperation] = {}
        self.connection_pools: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Metrics
        self.request_counter = Counter('sap_requests_total', 
                                     'Total SAP requests', 
                                     ['connection', 'operation', 'status'])
        self.request_duration = Histogram('sap_request_duration_seconds',
                                        'SAP request duration',
                                        ['connection', 'operation'])
        self.connection_pool_size = Gauge('sap_connection_pool_size',
                                        'SAP connection pool size',
                                        ['connection'])
        self.circuit_breaker_state = Gauge('sap_circuit_breaker_state',
                                         'SAP circuit breaker state (0=closed, 1=open)',
                                         ['connection'])
        
        logger.info("SAP Connector initialized")

    async def initialize(self):
        """Initialize the SAP Connector"""
        try:
            # Initialize Redis for caching and session management
            self.redis_client = aioredis.from_url(
                "redis://localhost:6379/8",
                decode_responses=True
            )
            
            # Load default connections
            await self._load_default_connections()
            
            # Initialize connection pools
            await self._initialize_connection_pools()
            
            # Initialize circuit breakers
            await self._initialize_circuit_breakers()
            
            logger.info("SAP Connector initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize SAP Connector: {e}")
            raise

    async def _load_default_connections(self):
        """Load default SAP connection configurations"""
        default_connections = [
            SAPConnection(
                connection_id="erp_production",
                name="ERP Production System",
                host="sap-erp-prod.company.com",
                port=8000,
                client="100",
                system_id="PRD",
                system_type=SAPSystemType.ERP,
                authentication=AuthenticationType.BASIC,
                username="API_USER",
                password="encrypted_password",
                pool_size=20,
                timeout_seconds=60
            ),
            SAPConnection(
                connection_id="s4_hana_dev",
                name="S/4 HANA Development",
                host="sap-s4-dev.company.com",
                port=44300,
                client="200",
                system_id="DEV",
                system_type=SAPSystemType.S4_HANA,
                authentication=AuthenticationType.OAUTH2,
                username="DEV_API_USER",
                password="dev_password",
                pool_size=10,
                timeout_seconds=30
            ),
            SAPConnection(
                connection_id="bw_analytics",
                name="BW Analytics System",
                host="sap-bw.company.com",
                port=8001,
                client="300",
                system_id="BWP",
                system_type=SAPSystemType.BW,
                authentication=AuthenticationType.BASIC,
                username="BW_USER",
                password="bw_password",
                pool_size=15
            )
        ]
        
        for connection in default_connections:
            self.connections[connection.connection_id] = connection
        
        # Load default operations
        default_operations = [
            SAPOperation(
                operation_id="get_customer_data",
                name="Get Customer Master Data",
                pattern=IntegrationPattern.BAPI,
                bapi_name="BAPI_CUSTOMER_GETDETAIL2",
                input_mapping={
                    "customer_number": "CUSTOMERNO",
                    "company_code": "COMPANYCODE"
                },
                output_mapping={
                    "customer_data": "CUSTOMERDETAIL",
                    "addresses": "CUSTOMERADDRESS"
                }
            ),
            SAPOperation(
                operation_id="create_sales_order",
                name="Create Sales Order",
                pattern=IntegrationPattern.BAPI,
                bapi_name="BAPI_SALESORDER_CREATEFROMDAT2",
                input_mapping={
                    "order_header": "ORDER_HEADER_IN",
                    "order_items": "ORDER_ITEMS_IN",
                    "partners": "ORDER_PARTNERS"
                },
                output_mapping={
                    "sales_document": "SALESDOCUMENT",
                    "return_messages": "RETURN"
                }
            ),
            SAPOperation(
                operation_id="get_material_info",
                name="Get Material Information",
                pattern=IntegrationPattern.RFC,
                function_module="RFC_READ_TABLE",
                input_mapping={
                    "table": "QUERY_TABLE",
                    "fields": "FIELDS",
                    "options": "OPTIONS"
                },
                output_mapping={
                    "data": "DATA",
                    "fields": "FIELDS"
                }
            ),
            SAPOperation(
                operation_id="send_idoc",
                name="Send IDoc to SAP",
                pattern=IntegrationPattern.IDOC,
                idoc_type="ORDERS05",
                input_mapping={
                    "control_record": "EDI_DC40",
                    "data_records": "EDI_DD40"
                }
            ),
            SAPOperation(
                operation_id="odata_products",
                name="OData Product Service",
                pattern=IntegrationPattern.ODATA,
                odata_service="/sap/opu/odata/sap/API_PRODUCT_SRV",
                endpoint_path="/A_Product",
                input_mapping={
                    "filters": "$filter",
                    "select": "$select",
                    "expand": "$expand"
                }
            )
        ]
        
        for operation in default_operations:
            self.operations[operation.operation_id] = operation
        
        logger.info(f"Loaded {len(default_connections)} connections and {len(default_operations)} operations")

    async def _initialize_connection_pools(self):
        """Initialize connection pools for each SAP system"""
        try:
            for connection_id, connection in self.connections.items():
                # For demo purposes, we'll simulate connection pools
                # In real implementation, this would use pyrfc or similar SAP connector
                self.connection_pools[connection_id] = {
                    "active_connections": 0,
                    "max_connections": connection.pool_size,
                    "available_connections": connection.pool_size,
                    "total_requests": 0,
                    "failed_requests": 0,
                    "last_used": datetime.utcnow()
                }
                
                # Update metrics
                self.connection_pool_size.labels(connection=connection_id).set(
                    connection.pool_size
                )
                
            logger.info("Connection pools initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise

    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for each connection"""
        for connection_id in self.connections.keys():
            self.circuit_breakers[connection_id] = {
                "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                "failure_count": 0,
                "failure_threshold": 5,
                "success_threshold": 3,
                "timeout_seconds": 60,
                "last_failure_time": None,
                "next_attempt_time": None
            }
            
            # Update metrics
            self.circuit_breaker_state.labels(connection=connection_id).set(0)

    async def register_connection(self, connection: SAPConnection) -> bool:
        """Register a new SAP connection"""
        try:
            self.connections[connection.connection_id] = connection
            
            # Initialize connection pool
            self.connection_pools[connection.connection_id] = {
                "active_connections": 0,
                "max_connections": connection.pool_size,
                "available_connections": connection.pool_size,
                "total_requests": 0,
                "failed_requests": 0,
                "last_used": datetime.utcnow()
            }
            
            # Initialize circuit breaker
            self.circuit_breakers[connection.connection_id] = {
                "state": "CLOSED",
                "failure_count": 0,
                "failure_threshold": 5,
                "success_threshold": 3,
                "timeout_seconds": 60,
                "last_failure_time": None,
                "next_attempt_time": None
            }
            
            logger.info(f"Registered SAP connection: {connection.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register connection {connection.connection_id}: {e}")
            return False

    async def register_operation(self, operation: SAPOperation) -> bool:
        """Register a new SAP operation"""
        try:
            self.operations[operation.operation_id] = operation
            logger.info(f"Registered SAP operation: {operation.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register operation {operation.operation_id}: {e}")
            return False

    async def execute_request(self, request: SAPRequest) -> SAPResponse:
        """Execute SAP request"""
        start_time = datetime.utcnow()
        
        try:
            # Validate connection and operation
            connection = self.connections.get(request.connection_id)
            operation = self.operations.get(request.operation_id)
            
            if not connection:
                return SAPResponse(
                    request_id=request.request_id,
                    success=False,
                    data=None,
                    error_message=f"Connection not found: {request.connection_id}"
                )
            
            if not operation:
                return SAPResponse(
                    request_id=request.request_id,
                    success=False,
                    data=None,
                    error_message=f"Operation not found: {request.operation_id}"
                )
            
            # Check circuit breaker
            circuit_state = await self._check_circuit_breaker(request.connection_id)
            if circuit_state == "OPEN":
                return SAPResponse(
                    request_id=request.request_id,
                    success=False,
                    data=None,
                    error_message="Circuit breaker is OPEN - service temporarily unavailable"
                )
            
            # Execute based on integration pattern
            if operation.pattern == IntegrationPattern.BAPI:
                response = await self._execute_bapi(request, connection, operation)
            elif operation.pattern == IntegrationPattern.RFC:
                response = await self._execute_rfc(request, connection, operation)
            elif operation.pattern == IntegrationPattern.IDOC:
                response = await self._execute_idoc(request, connection, operation)
            elif operation.pattern == IntegrationPattern.ODATA:
                response = await self._execute_odata(request, connection, operation)
            elif operation.pattern == IntegrationPattern.REST_API:
                response = await self._execute_rest_api(request, connection, operation)
            elif operation.pattern == IntegrationPattern.SOAP:
                response = await self._execute_soap(request, connection, operation)
            else:
                response = SAPResponse(
                    request_id=request.request_id,
                    success=False,
                    data=None,
                    error_message=f"Unsupported integration pattern: {operation.pattern.value}"
                )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            response.execution_time_ms = int(execution_time)
            
            # Update circuit breaker
            if response.success:
                await self._record_circuit_breaker_success(request.connection_id)
            else:
                await self._record_circuit_breaker_failure(request.connection_id)
            
            # Update metrics
            status = "success" if response.success else "error"
            self.request_counter.labels(
                connection=request.connection_id,
                operation=request.operation_id,
                status=status
            ).inc()
            
            self.request_duration.labels(
                connection=request.connection_id,
                operation=request.operation_id
            ).observe(execution_time / 1000)
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing SAP request {request.request_id}: {e}")
            
            # Record failure
            await self._record_circuit_breaker_failure(request.connection_id)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return SAPResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=str(e),
                execution_time_ms=int(execution_time)
            )

    async def _execute_bapi(self, request: SAPRequest, connection: SAPConnection, 
                          operation: SAPOperation) -> SAPResponse:
        """Execute BAPI call"""
        try:
            # Map input parameters
            mapped_params = await self._map_input_parameters(
                request.parameters, operation.input_mapping
            )
            
            # Simulate BAPI call
            # In real implementation, this would use pyrfc.Connection
            bapi_result = await self._simulate_bapi_call(
                connection, operation.bapi_name, mapped_params
            )
            
            # Map output parameters
            if operation.output_mapping and bapi_result.get('success'):
                mapped_output = await self._map_output_parameters(
                    bapi_result['data'], operation.output_mapping
                )
                bapi_result['data'] = mapped_output
            
            return SAPResponse(
                request_id=request.request_id,
                success=bapi_result.get('success', False),
                data=bapi_result.get('data'),
                error_message=bapi_result.get('error'),
                sap_return_code=bapi_result.get('return_code'),
                warnings=bapi_result.get('warnings', [])
            )
            
        except Exception as e:
            logger.error(f"BAPI execution error: {e}")
            return SAPResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=f"BAPI execution failed: {e}"
            )

    async def _execute_rfc(self, request: SAPRequest, connection: SAPConnection,
                         operation: SAPOperation) -> SAPResponse:
        """Execute RFC function module"""
        try:
            # Map input parameters
            mapped_params = await self._map_input_parameters(
                request.parameters, operation.input_mapping
            )
            
            # Simulate RFC call
            rfc_result = await self._simulate_rfc_call(
                connection, operation.function_module, mapped_params
            )
            
            # Map output parameters
            if operation.output_mapping and rfc_result.get('success'):
                mapped_output = await self._map_output_parameters(
                    rfc_result['data'], operation.output_mapping
                )
                rfc_result['data'] = mapped_output
            
            return SAPResponse(
                request_id=request.request_id,
                success=rfc_result.get('success', False),
                data=rfc_result.get('data'),
                error_message=rfc_result.get('error'),
                sap_return_code=rfc_result.get('return_code')
            )
            
        except Exception as e:
            logger.error(f"RFC execution error: {e}")
            return SAPResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=f"RFC execution failed: {e}"
            )

    async def _execute_idoc(self, request: SAPRequest, connection: SAPConnection,
                          operation: SAPOperation) -> SAPResponse:
        """Execute IDoc processing"""
        try:
            # Build IDoc structure
            idoc_data = await self._build_idoc_structure(
                request.parameters, operation
            )
            
            # Send IDoc
            idoc_result = await self._send_idoc(connection, idoc_data)
            
            return SAPResponse(
                request_id=request.request_id,
                success=idoc_result.get('success', False),
                data=idoc_result.get('data'),
                error_message=idoc_result.get('error'),
                sap_return_code=idoc_result.get('return_code')
            )
            
        except Exception as e:
            logger.error(f"IDoc execution error: {e}")
            return SAPResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=f"IDoc processing failed: {e}"
            )

    async def _execute_odata(self, request: SAPRequest, connection: SAPConnection,
                           operation: SAPOperation) -> SAPResponse:
        """Execute OData service call"""
        try:
            # Build OData URL
            base_url = f"https://{connection.host}:{connection.port}{operation.odata_service}"
            endpoint_url = f"{base_url}{operation.endpoint_path}"
            
            # Build query parameters
            query_params = {}
            if operation.input_mapping:
                for param_key, param_value in request.parameters.items():
                    if param_key in operation.input_mapping:
                        odata_param = operation.input_mapping[param_key]
                        query_params[odata_param] = param_value
            
            # Execute OData request
            odata_result = await self._execute_odata_request(
                connection, endpoint_url, query_params, request.headers or {}
            )
            
            return SAPResponse(
                request_id=request.request_id,
                success=odata_result.get('success', False),
                data=odata_result.get('data'),
                error_message=odata_result.get('error')
            )
            
        except Exception as e:
            logger.error(f"OData execution error: {e}")
            return SAPResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=f"OData execution failed: {e}"
            )

    async def _execute_rest_api(self, request: SAPRequest, connection: SAPConnection,
                              operation: SAPOperation) -> SAPResponse:
        """Execute REST API call"""
        try:
            # Build REST URL
            base_url = f"https://{connection.host}:{connection.port}"
            endpoint_url = f"{base_url}{operation.endpoint_path}"
            
            # Execute REST request
            rest_result = await self._execute_rest_request(
                connection, endpoint_url, request.parameters, request.headers or {}
            )
            
            return SAPResponse(
                request_id=request.request_id,
                success=rest_result.get('success', False),
                data=rest_result.get('data'),
                error_message=rest_result.get('error')
            )
            
        except Exception as e:
            logger.error(f"REST API execution error: {e}")
            return SAPResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=f"REST API execution failed: {e}"
            )

    async def _execute_soap(self, request: SAPRequest, connection: SAPConnection,
                          operation: SAPOperation) -> SAPResponse:
        """Execute SOAP web service call"""
        try:
            # Build SOAP envelope
            soap_envelope = await self._build_soap_envelope(
                request.parameters, operation
            )
            
            # Execute SOAP request
            soap_result = await self._execute_soap_request(
                connection, operation.endpoint_path, soap_envelope, request.headers or {}
            )
            
            return SAPResponse(
                request_id=request.request_id,
                success=soap_result.get('success', False),
                data=soap_result.get('data'),
                error_message=soap_result.get('error')
            )
            
        except Exception as e:
            logger.error(f"SOAP execution error: {e}")
            return SAPResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=f"SOAP execution failed: {e}"
            )

    async def _simulate_bapi_call(self, connection: SAPConnection, 
                                bapi_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate BAPI call (replace with actual pyrfc implementation)"""
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # Simulate different BAPI responses
        if bapi_name == "BAPI_CUSTOMER_GETDETAIL2":
            return {
                "success": True,
                "data": {
                    "CUSTOMERDETAIL": {
                        "CUSTOMER": parameters.get("CUSTOMERNO", ""),
                        "NAME": "Sample Customer Name",
                        "COUNTRY": "US",
                        "CITY": "New York"
                    },
                    "CUSTOMERADDRESS": [
                        {
                            "ADDRESS_TYPE": "BILL_TO",
                            "STREET": "123 Main St",
                            "POSTAL_CODE": "10001"
                        }
                    ]
                },
                "return_code": "S"
            }
        elif bapi_name == "BAPI_SALESORDER_CREATEFROMDAT2":
            return {
                "success": True,
                "data": {
                    "SALESDOCUMENT": "0000012345",
                    "RETURN": [
                        {
                            "TYPE": "S",
                            "MESSAGE": "Sales order created successfully"
                        }
                    ]
                },
                "return_code": "S"
            }
        else:
            return {
                "success": False,
                "error": f"Unknown BAPI: {bapi_name}",
                "return_code": "E"
            }

    async def _simulate_rfc_call(self, connection: SAPConnection,
                               function_module: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate RFC function call"""
        await asyncio.sleep(0.05)  # Simulate network latency
        
        if function_module == "RFC_READ_TABLE":
            table_name = parameters.get("QUERY_TABLE", "")
            return {
                "success": True,
                "data": {
                    "DATA": [
                        {"WA": "SAMPLE_DATA_ROW_1"},
                        {"WA": "SAMPLE_DATA_ROW_2"}
                    ],
                    "FIELDS": [
                        {"FIELDNAME": "FIELD1", "TYPE": "C", "LENGTH": 10},
                        {"FIELDNAME": "FIELD2", "TYPE": "N", "LENGTH": 8}
                    ]
                },
                "return_code": "0"
            }
        else:
            return {
                "success": False,
                "error": f"Unknown function module: {function_module}",
                "return_code": "4"
            }

    async def _build_idoc_structure(self, parameters: Dict[str, Any], 
                                  operation: SAPOperation) -> Dict[str, Any]:
        """Build IDoc structure"""
        control_record = {
            "TABNAM": "EDI_DC40",
            "DOCNUM": f"IDOC_{int(datetime.utcnow().timestamp())}",
            "IDOCTYP": operation.idoc_type,
            "MESTYP": "ORDERS",
            "RCVPRT": "LS",
            "RCVPRN": "PARTNER_SYSTEM",
            "SNDPRT": "LS",
            "SNDPRN": "SOURCE_SYSTEM"
        }
        
        data_records = parameters.get("data_records", [])
        
        return {
            "control_record": control_record,
            "data_records": data_records
        }

    async def _send_idoc(self, connection: SAPConnection, 
                       idoc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send IDoc to SAP system"""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        return {
            "success": True,
            "data": {
                "idoc_number": idoc_data["control_record"]["DOCNUM"],
                "status": "03",  # Processed successfully
                "message": "IDoc processed successfully"
            },
            "return_code": "0"
        }

    async def _execute_odata_request(self, connection: SAPConnection, url: str,
                                   params: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Execute OData HTTP request"""
        try:
            timeout = aiohttp.ClientTimeout(total=connection.timeout_seconds)
            auth = aiohttp.BasicAuth(connection.username, connection.password)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers, auth=auth) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        # Parse OData response (usually JSON or XML)
                        try:
                            data = json.loads(response_text)
                        except json.JSONDecodeError:
                            # Try XML parsing for older OData versions
                            data = xmltodict.parse(response_text)
                        
                        return {
                            "success": True,
                            "data": data
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {response_text}"
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"OData request failed: {e}"
            }

    async def _execute_rest_request(self, connection: SAPConnection, url: str,
                                  data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Execute REST API request"""
        try:
            timeout = aiohttp.ClientTimeout(total=connection.timeout_seconds)
            auth = aiohttp.BasicAuth(connection.username, connection.password)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=data, headers=headers, auth=auth) as response:
                    response_text = await response.text()
                    
                    if response.status in [200, 201]:
                        try:
                            response_data = json.loads(response_text)
                        except json.JSONDecodeError:
                            response_data = response_text
                        
                        return {
                            "success": True,
                            "data": response_data
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {response_text}"
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"REST request failed: {e}"
            }

    async def _build_soap_envelope(self, parameters: Dict[str, Any], 
                                 operation: SAPOperation) -> str:
        """Build SOAP envelope"""
        # Simple SOAP envelope template
        envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"
               xmlns:web="http://company.com/webservice">
    <soap:Header/>
    <soap:Body>
        <web:ExecuteOperation>
            {self._dict_to_xml_elements(parameters)}
        </web:ExecuteOperation>
    </soap:Body>
</soap:Envelope>"""
        
        return envelope

    def _dict_to_xml_elements(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to XML elements"""
        elements = []
        for key, value in data.items():
            if isinstance(value, dict):
                elements.append(f"<{key}>{self._dict_to_xml_elements(value)}</{key}>")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        elements.append(f"<{key}>{self._dict_to_xml_elements(item)}</{key}>")
                    else:
                        elements.append(f"<{key}>{item}</{key}>")
            else:
                elements.append(f"<{key}>{value}</{key}>")
        
        return ''.join(elements)

    async def _execute_soap_request(self, connection: SAPConnection, endpoint: str,
                                  envelope: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Execute SOAP request"""
        try:
            url = f"https://{connection.host}:{connection.port}{endpoint}"
            soap_headers = {
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": '""',
                **headers
            }
            
            timeout = aiohttp.ClientTimeout(total=connection.timeout_seconds)
            auth = aiohttp.BasicAuth(connection.username, connection.password)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=envelope, headers=soap_headers, auth=auth) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        # Parse SOAP response
                        try:
                            root = ET.fromstring(response_text)
                            # Extract body content (simplified)
                            body = root.find('.//{http://schemas.xmlsoap.org/soap/envelope/}Body')
                            if body is not None:
                                # Convert to dict
                                data = self._xml_element_to_dict(body)
                            else:
                                data = response_text
                        except ET.ParseError:
                            data = response_text
                        
                        return {
                            "success": True,
                            "data": data
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"SOAP fault: HTTP {response.status}: {response_text}"
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"SOAP request failed: {e}"
            }

    def _xml_element_to_dict(self, element) -> Dict[str, Any]:
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
            child_data = self._xml_element_to_dict(child)
            tag = child.tag.split('}')[-1]  # Remove namespace
            
            if tag in result:
                # Convert to list if multiple elements with same tag
                if not isinstance(result[tag], list):
                    result[tag] = [result[tag]]
                result[tag].append(child_data)
            else:
                result[tag] = child_data
        
        return result if result else None

    async def _map_input_parameters(self, parameters: Dict[str, Any], 
                                  mapping: Optional[Dict[str, str]]) -> Dict[str, Any]:
        """Map input parameters according to mapping rules"""
        if not mapping:
            return parameters
        
        mapped_params = {}
        for source_key, target_key in mapping.items():
            if source_key in parameters:
                mapped_params[target_key] = parameters[source_key]
        
        # Include unmapped parameters
        for key, value in parameters.items():
            if key not in mapping and key not in mapped_params:
                mapped_params[key] = value
        
        return mapped_params

    async def _map_output_parameters(self, data: Any, 
                                   mapping: Optional[Dict[str, str]]) -> Any:
        """Map output parameters according to mapping rules"""
        if not mapping or not isinstance(data, dict):
            return data
        
        mapped_data = {}
        for source_key, target_key in mapping.items():
            if source_key in data:
                mapped_data[target_key] = data[source_key]
        
        return mapped_data

    async def _check_circuit_breaker(self, connection_id: str) -> str:
        """Check circuit breaker state"""
        try:
            breaker = self.circuit_breakers.get(connection_id)
            if not breaker:
                return "CLOSED"
            
            current_time = datetime.utcnow()
            
            if breaker["state"] == "OPEN":
                if (breaker["next_attempt_time"] and 
                    current_time >= breaker["next_attempt_time"]):
                    breaker["state"] = "HALF_OPEN"
                    logger.info(f"Circuit breaker for {connection_id} moved to HALF_OPEN")
                    return "HALF_OPEN"
                return "OPEN"
            
            return breaker["state"]
            
        except Exception as e:
            logger.error(f"Circuit breaker check error: {e}")
            return "CLOSED"

    async def _record_circuit_breaker_success(self, connection_id: str):
        """Record successful request for circuit breaker"""
        try:
            breaker = self.circuit_breakers.get(connection_id)
            if not breaker:
                return
            
            if breaker["state"] == "HALF_OPEN":
                breaker["failure_count"] = 0
                breaker["state"] = "CLOSED"
                self.circuit_breaker_state.labels(connection=connection_id).set(0)
                logger.info(f"Circuit breaker for {connection_id} closed after successful request")
            elif breaker["state"] == "CLOSED":
                breaker["failure_count"] = 0
                
        except Exception as e:
            logger.error(f"Circuit breaker success recording error: {e}")

    async def _record_circuit_breaker_failure(self, connection_id: str):
        """Record failed request for circuit breaker"""
        try:
            breaker = self.circuit_breakers.get(connection_id)
            if not breaker:
                return
            
            breaker["failure_count"] += 1
            breaker["last_failure_time"] = datetime.utcnow()
            
            if breaker["failure_count"] >= breaker["failure_threshold"]:
                breaker["state"] = "OPEN"
                breaker["next_attempt_time"] = datetime.utcnow() + timedelta(
                    seconds=breaker["timeout_seconds"]
                )
                
                self.circuit_breaker_state.labels(connection=connection_id).set(1)
                logger.warning(f"Circuit breaker for {connection_id} opened after {breaker['failure_count']} failures")
                
        except Exception as e:
            logger.error(f"Circuit breaker failure recording error: {e}")

    async def get_connection_health(self) -> Dict[str, Any]:
        """Get health status of all SAP connections"""
        try:
            health_status = {}
            
            for connection_id, connection in self.connections.items():
                pool = self.connection_pools.get(connection_id, {})
                breaker = self.circuit_breakers.get(connection_id, {})
                
                health_status[connection_id] = {
                    "name": connection.name,
                    "host": connection.host,
                    "system_type": connection.system_type.value,
                    "circuit_breaker_state": breaker.get("state", "UNKNOWN"),
                    "failure_count": breaker.get("failure_count", 0),
                    "active_connections": pool.get("active_connections", 0),
                    "total_requests": pool.get("total_requests", 0),
                    "failed_requests": pool.get("failed_requests", 0),
                    "last_used": pool.get("last_used", datetime.utcnow()).isoformat()
                }
            
            return {
                "status": "healthy",
                "connections": health_status,
                "total_operations": len(self.operations),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for SAP connections"""
        try:
            metrics = {}
            
            for connection_id in self.connections.keys():
                pool = self.connection_pools.get(connection_id, {})
                breaker = self.circuit_breakers.get(connection_id, {})
                
                metrics[connection_id] = {
                    "total_requests": pool.get("total_requests", 0),
                    "failed_requests": pool.get("failed_requests", 0),
                    "success_rate": (
                        (pool.get("total_requests", 0) - pool.get("failed_requests", 0)) / 
                        max(pool.get("total_requests", 1), 1) * 100
                    ),
                    "circuit_breaker_state": breaker.get("state", "UNKNOWN"),
                    "active_connections": pool.get("active_connections", 0),
                    "available_connections": pool.get("available_connections", 0)
                }
            
            return {
                "connections": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("SAP Connector cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Example usage and testing
async def main():
    connector = SAPConnector()
    await connector.initialize()
    
    # Test BAPI request
    bapi_request = SAPRequest(
        request_id="test_bapi_001",
        connection_id="erp_production",
        operation_id="get_customer_data",
        parameters={
            "customer_number": "0000100001",
            "company_code": "1000"
        }
    )
    
    response = await connector.execute_request(bapi_request)
    print(f"BAPI Response: {json.dumps(asdict(response), indent=2, default=str)}")
    
    # Test OData request
    odata_request = SAPRequest(
        request_id="test_odata_001",
        connection_id="s4_hana_dev",
        operation_id="odata_products",
        parameters={
            "filters": "ProductType eq 'FERT'",
            "select": "Product,ProductType,ProductDescription"
        }
    )
    
    response = await connector.execute_request(odata_request)
    print(f"OData Response: {json.dumps(asdict(response), indent=2, default=str)}")
    
    # Get health status
    health = await connector.get_connection_health()
    print(f"Health: {json.dumps(health, indent=2)}")
    
    await connector.cleanup()

if __name__ == "__main__":
    asyncio.run(main())