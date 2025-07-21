"""
Webhook and Event System for Pynomaly Detection
===============================================

Comprehensive event-driven system providing:
- Webhook registration and management
- Event publishing and subscription
- Real-time notifications and alerts
- Message queuing and delivery guarantees
- Event filtering and routing
- Retry mechanisms and error handling
- Performance monitoring and analytics
"""

import logging
import json
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import uuid
import hashlib
import hmac

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event type enumeration."""
    ANOMALY_DETECTED = "anomaly_detected"
    MODEL_TRAINED = "model_trained"
    DATASET_UPLOADED = "dataset_uploaded"
    USER_REGISTERED = "user_registered"
    PROJECT_CREATED = "project_created"
    EXTENSION_INSTALLED = "extension_installed"
    SYSTEM_ALERT = "system_alert"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    INTEGRATION_FAILED = "integration_failed"
    CUSTOM = "custom"

class DeliveryMethod(Enum):
    """Event delivery method enumeration."""
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    PUSH_NOTIFICATION = "push_notification"
    INTERNAL = "internal"

class EventPriority(Enum):
    """Event priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class DeliveryStatus(Enum):
    """Delivery status enumeration."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"

@dataclass
class Event:
    """Event definition."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    expires_at: Optional[datetime] = None

@dataclass
class WebhookEndpoint:
    """Webhook endpoint definition."""
    webhook_id: str
    name: str
    url: str
    secret: str
    event_types: List[EventType] = field(default_factory=list)
    delivery_method: DeliveryMethod = DeliveryMethod.WEBHOOK
    is_active: bool = True
    retry_attempts: int = 3
    timeout_seconds: int = 30
    filters: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)
    last_delivery: Optional[datetime] = None
    delivery_count: int = 0
    failure_count: int = 0
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class EventSubscription:
    """Event subscription definition."""
    subscription_id: str
    webhook_id: str
    event_types: List[EventType] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None

@dataclass
class DeliveryAttempt:
    """Delivery attempt record."""
    attempt_id: str
    event_id: str
    webhook_id: str
    attempt_number: int
    timestamp: datetime
    status: DeliveryStatus
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: float = 0

class WebhookManager:
    """Comprehensive webhook management system."""
    
    def __init__(self):
        """Initialize webhook manager."""
        # Webhook storage
        self.webhooks: Dict[str, WebhookEndpoint] = {}
        self.subscriptions: Dict[str, EventSubscription] = {}
        
        # Event queue and delivery
        self.event_queue: deque = deque()
        self.delivery_attempts: List[DeliveryAttempt] = []
        
        # Retry queue for failed deliveries
        self.retry_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Threading and async
        self.lock = threading.RLock()
        self.delivery_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'active_webhooks': 0,
            'avg_delivery_time': 0.0
        }
        
        # Start delivery worker
        self._start_delivery_worker()
        
        logger.info("Webhook Manager initialized")
    
    def register_webhook(self, webhook_data: Dict[str, Any]) -> str:
        """Register new webhook endpoint.
        
        Args:
            webhook_data: Webhook configuration
            
        Returns:
            Webhook ID
        """
        try:
            with self.lock:
                webhook_id = str(uuid.uuid4())
                
                # Generate secret if not provided
                secret = webhook_data.get('secret')
                if not secret:
                    secret = self._generate_webhook_secret()
                
                webhook = WebhookEndpoint(
                    webhook_id=webhook_id,
                    name=webhook_data['name'],
                    url=webhook_data['url'],
                    secret=secret,
                    event_types=[EventType(et) for et in webhook_data.get('event_types', [])],
                    delivery_method=DeliveryMethod(webhook_data.get('delivery_method', 'webhook')),
                    retry_attempts=webhook_data.get('retry_attempts', 3),
                    timeout_seconds=webhook_data.get('timeout_seconds', 30),
                    filters=webhook_data.get('filters', {}),
                    headers=webhook_data.get('headers', {}),
                    tenant_id=webhook_data.get('tenant_id'),
                    user_id=webhook_data.get('user_id')
                )
                
                self.webhooks[webhook_id] = webhook
                self.stats['active_webhooks'] += 1
                
                logger.info(f"Webhook registered: {webhook.name} ({webhook_id})")
                return webhook_id
                
        except Exception as e:
            logger.error(f"Webhook registration failed: {e}")
            raise
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister webhook endpoint.
        
        Args:
            webhook_id: Webhook identifier
            
        Returns:
            True if unregistration successful
        """
        try:
            with self.lock:
                if webhook_id not in self.webhooks:
                    logger.warning(f"Webhook not found: {webhook_id}")
                    return False
                
                # Remove webhook
                webhook = self.webhooks[webhook_id]
                del self.webhooks[webhook_id]
                
                # Remove related subscriptions
                to_remove = [sid for sid, sub in self.subscriptions.items() 
                           if sub.webhook_id == webhook_id]
                for sid in to_remove:
                    del self.subscriptions[sid]
                
                # Update statistics
                if self.stats['active_webhooks'] > 0:
                    self.stats['active_webhooks'] -= 1
                
                logger.info(f"Webhook unregistered: {webhook.name}")
                return True
                
        except Exception as e:
            logger.error(f"Webhook unregistration failed: {e}")
            return False
    
    def subscribe_to_events(self, webhook_id: str, event_types: List[str],
                           filters: Optional[Dict[str, Any]] = None) -> str:
        """Subscribe webhook to specific events.
        
        Args:
            webhook_id: Webhook identifier
            event_types: List of event types to subscribe to
            filters: Optional event filters
            
        Returns:
            Subscription ID
        """
        try:
            with self.lock:
                if webhook_id not in self.webhooks:
                    raise ValueError(f"Webhook not found: {webhook_id}")
                
                subscription_id = str(uuid.uuid4())
                webhook = self.webhooks[webhook_id]
                
                subscription = EventSubscription(
                    subscription_id=subscription_id,
                    webhook_id=webhook_id,
                    event_types=[EventType(et) for et in event_types],
                    filters=filters or {},
                    tenant_id=webhook.tenant_id,
                    user_id=webhook.user_id
                )
                
                self.subscriptions[subscription_id] = subscription
                
                logger.info(f"Event subscription created: {subscription_id}")
                return subscription_id
                
        except Exception as e:
            logger.error(f"Event subscription failed: {e}")
            raise
    
    def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """Unsubscribe from events.
        
        Args:
            subscription_id: Subscription identifier
            
        Returns:
            True if unsubscription successful
        """
        try:
            with self.lock:
                if subscription_id not in self.subscriptions:
                    logger.warning(f"Subscription not found: {subscription_id}")
                    return False
                
                del self.subscriptions[subscription_id]
                
                logger.info(f"Event subscription removed: {subscription_id}")
                return True
                
        except Exception as e:
            logger.error(f"Event unsubscription failed: {e}")
            return False
    
    def publish_event(self, event: Event) -> bool:
        """Publish event to subscribers.
        
        Args:
            event: Event to publish
            
        Returns:
            True if event queued successfully
        """
        try:
            with self.lock:
                # Add to event queue
                self.event_queue.append(event)
                self.stats['total_events'] += 1
                
                logger.debug(f"Event published: {event.event_type.value} ({event.event_id})")
                return True
                
        except Exception as e:
            logger.error(f"Event publishing failed: {e}")
            return False
    
    def get_webhook_status(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook status information.
        
        Args:
            webhook_id: Webhook identifier
            
        Returns:
            Webhook status or None
        """
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return None
        
        # Get recent delivery attempts
        recent_attempts = [
            attempt for attempt in self.delivery_attempts[-50:]
            if attempt.webhook_id == webhook_id
        ]
        
        # Calculate success rate
        if recent_attempts:
            successful = sum(1 for a in recent_attempts if a.status == DeliveryStatus.DELIVERED)
            success_rate = (successful / len(recent_attempts)) * 100
        else:
            success_rate = 0.0
        
        return {
            'webhook_id': webhook_id,
            'name': webhook.name,
            'is_active': webhook.is_active,
            'delivery_count': webhook.delivery_count,
            'failure_count': webhook.failure_count,
            'last_delivery': webhook.last_delivery.isoformat() if webhook.last_delivery else None,
            'success_rate': success_rate,
            'recent_attempts': len(recent_attempts)
        }
    
    def get_delivery_history(self, webhook_id: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get delivery history.
        
        Args:
            webhook_id: Optional webhook filter
            limit: Maximum number of attempts to return
            
        Returns:
            List of delivery attempts
        """
        attempts = self.delivery_attempts
        
        if webhook_id:
            attempts = [a for a in attempts if a.webhook_id == webhook_id]
        
        # Sort by timestamp (most recent first) and limit
        attempts = sorted(attempts, key=lambda a: a.timestamp, reverse=True)[:limit]
        
        return [
            {
                'attempt_id': attempt.attempt_id,
                'event_id': attempt.event_id,
                'webhook_id': attempt.webhook_id,
                'attempt_number': attempt.attempt_number,
                'timestamp': attempt.timestamp.isoformat(),
                'status': attempt.status.value,
                'response_code': attempt.response_code,
                'duration_ms': attempt.duration_ms,
                'error_message': attempt.error_message
            }
            for attempt in attempts
        ]
    
    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Test webhook endpoint.
        
        Args:
            webhook_id: Webhook identifier
            
        Returns:
            Test result
        """
        try:
            webhook = self.webhooks.get(webhook_id)
            if not webhook:
                return {'success': False, 'error': 'Webhook not found'}
            
            # Create test event
            test_event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.CUSTOM,
                timestamp=datetime.now(),
                source='webhook_test',
                data={'message': 'This is a test event'},
                metadata={'test': True}
            )
            
            # Attempt delivery
            result = self._deliver_event(webhook, test_event)
            
            return {
                'success': result.status == DeliveryStatus.DELIVERED,
                'status': result.status.value,
                'response_code': result.response_code,
                'duration_ms': result.duration_ms,
                'error_message': result.error_message
            }
            
        except Exception as e:
            logger.error(f"Webhook test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get webhook system statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate additional metrics
            stats.update({
                'total_webhooks': len(self.webhooks),
                'total_subscriptions': len(self.subscriptions),
                'queue_length': len(self.event_queue),
                'retry_queue_length': sum(len(items) for items in self.retry_queue.values()),
                'recent_deliveries': len([a for a in self.delivery_attempts 
                                        if (datetime.now() - a.timestamp).seconds < 3600])
            })
            
            # Event type distribution
            event_type_counts = defaultdict(int)
            for attempt in self.delivery_attempts[-1000:]:  # Last 1000 attempts
                # Would need to track event type in delivery attempts
                pass
            
            return stats
    
    def shutdown(self):
        """Shutdown webhook manager."""
        logger.info("Shutting down webhook manager...")
        
        self.shutdown_event.set()
        
        if self.delivery_thread and self.delivery_thread.is_alive():
            self.delivery_thread.join(timeout=5)
        
        logger.info("Webhook manager shutdown complete")
    
    def _start_delivery_worker(self):
        """Start background delivery worker thread."""
        self.delivery_thread = threading.Thread(
            target=self._delivery_worker,
            name="webhook-delivery-worker",
            daemon=True
        )
        self.delivery_thread.start()
    
    def _delivery_worker(self):
        """Background worker for event delivery."""
        logger.info("Webhook delivery worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Process event queue
                events_to_process = []
                
                with self.lock:
                    # Get events from queue (up to 10 at a time)
                    for _ in range(min(10, len(self.event_queue))):
                        if self.event_queue:
                            events_to_process.append(self.event_queue.popleft())
                
                # Process events
                for event in events_to_process:
                    self._process_event(event)
                
                # Process retry queue
                self._process_retry_queue()
                
                # Sleep briefly if no events
                if not events_to_process:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Delivery worker error: {e}")
                time.sleep(1)
        
        logger.info("Webhook delivery worker stopped")
    
    def _process_event(self, event: Event):
        """Process single event for delivery.
        
        Args:
            event: Event to process
        """
        try:
            # Find matching subscriptions
            matching_webhooks = self._find_matching_webhooks(event)
            
            # Deliver to each matching webhook
            for webhook in matching_webhooks:
                if webhook.is_active:
                    self._schedule_delivery(webhook, event)
                    
        except Exception as e:
            logger.error(f"Event processing failed: {e}")
    
    def _find_matching_webhooks(self, event: Event) -> List[WebhookEndpoint]:
        """Find webhooks that should receive this event.
        
        Args:
            event: Event to match
            
        Returns:
            List of matching webhooks
        """
        matching_webhooks = []
        
        with self.lock:
            for subscription in self.subscriptions.values():
                if not subscription.is_active:
                    continue
                
                webhook = self.webhooks.get(subscription.webhook_id)
                if not webhook or not webhook.is_active:
                    continue
                
                # Check event type match
                if (not subscription.event_types or 
                    event.event_type in subscription.event_types):
                    
                    # Check filters
                    if self._event_matches_filters(event, subscription.filters):
                        
                        # Check tenant/user isolation
                        if self._check_tenant_access(event, webhook):
                            matching_webhooks.append(webhook)
        
        return matching_webhooks
    
    def _event_matches_filters(self, event: Event, filters: Dict[str, Any]) -> bool:
        """Check if event matches subscription filters.
        
        Args:
            event: Event to check
            filters: Filter criteria
            
        Returns:
            True if event matches filters
        """
        if not filters:
            return True
        
        try:
            # Priority filter
            if 'priority' in filters:
                if event.priority.value not in filters['priority']:
                    return False
            
            # Source filter
            if 'source' in filters:
                if event.source not in filters['source']:
                    return False
            
            # Tag filters
            if 'tags' in filters:
                required_tags = set(filters['tags'])
                event_tags = set(event.tags)
                if not required_tags.issubset(event_tags):
                    return False
            
            # Data field filters
            if 'data' in filters:
                for field, expected_value in filters['data'].items():
                    if field not in event.data or event.data[field] != expected_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Filter matching failed: {e}")
            return False
    
    def _check_tenant_access(self, event: Event, webhook: WebhookEndpoint) -> bool:
        """Check tenant/user access control.
        
        Args:
            event: Event to check
            webhook: Webhook endpoint
            
        Returns:
            True if access allowed
        """
        # Check tenant isolation
        if webhook.tenant_id and event.tenant_id:
            return webhook.tenant_id == event.tenant_id
        
        # Check user access (for user-specific webhooks)
        if webhook.user_id and event.user_id:
            return webhook.user_id == event.user_id
        
        # Allow if no restrictions or restrictions don't apply
        return True
    
    def _schedule_delivery(self, webhook: WebhookEndpoint, event: Event, attempt_number: int = 1):
        """Schedule event delivery to webhook.
        
        Args:
            webhook: Webhook endpoint
            event: Event to deliver
            attempt_number: Delivery attempt number
        """
        try:
            # Check if event has expired
            if event.expires_at and datetime.now() > event.expires_at:
                logger.debug(f"Event expired, skipping delivery: {event.event_id}")
                return
            
            # Perform delivery
            result = self._deliver_event(webhook, event, attempt_number)
            
            # Record attempt
            self.delivery_attempts.append(result)
            
            # Update webhook statistics
            with self.lock:
                webhook.delivery_count += 1
                webhook.last_delivery = datetime.now()
                
                if result.status == DeliveryStatus.DELIVERED:
                    self.stats['successful_deliveries'] += 1
                else:
                    webhook.failure_count += 1
                    self.stats['failed_deliveries'] += 1
                    
                    # Schedule retry if appropriate
                    if (attempt_number < webhook.retry_attempts and 
                        result.status == DeliveryStatus.FAILED):
                        self._schedule_retry(webhook, event, attempt_number + 1)
            
            # Update average delivery time
            if result.duration_ms > 0:
                self.stats['avg_delivery_time'] = (
                    self.stats['avg_delivery_time'] * 0.9 + result.duration_ms * 0.1
                )
                
        except Exception as e:
            logger.error(f"Delivery scheduling failed: {e}")
    
    def _deliver_event(self, webhook: WebhookEndpoint, event: Event, 
                      attempt_number: int = 1) -> DeliveryAttempt:
        """Deliver event to webhook endpoint.
        
        Args:
            webhook: Webhook endpoint
            event: Event to deliver
            attempt_number: Delivery attempt number
            
        Returns:
            Delivery attempt result
        """
        start_time = time.time()
        attempt_id = str(uuid.uuid4())
        
        attempt = DeliveryAttempt(
            attempt_id=attempt_id,
            event_id=event.event_id,
            webhook_id=webhook.webhook_id,
            attempt_number=attempt_number,
            timestamp=datetime.now(),
            status=DeliveryStatus.PENDING
        )
        
        try:
            if webhook.delivery_method == DeliveryMethod.WEBHOOK:
                result = self._deliver_webhook(webhook, event)
            else:
                # Handle other delivery methods
                result = self._deliver_alternative(webhook, event)
            
            attempt.status = DeliveryStatus.DELIVERED if result['success'] else DeliveryStatus.FAILED
            attempt.response_code = result.get('status_code')
            attempt.response_body = result.get('response_body', '')[:1000]  # Limit size
            attempt.error_message = result.get('error')
            
        except Exception as e:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = str(e)
            logger.error(f"Event delivery failed: {e}")
        
        finally:
            attempt.duration_ms = (time.time() - start_time) * 1000
        
        return attempt
    
    def _deliver_webhook(self, webhook: WebhookEndpoint, event: Event) -> Dict[str, Any]:
        """Deliver event via webhook.
        
        Args:
            webhook: Webhook endpoint
            event: Event to deliver
            
        Returns:
            Delivery result
        """
        if not REQUESTS_AVAILABLE:
            return {'success': False, 'error': 'HTTP client not available'}
        
        try:
            # Prepare payload
            payload = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'source': event.source,
                'data': event.data,
                'metadata': event.metadata,
                'priority': event.priority.value,
                'tenant_id': event.tenant_id,
                'user_id': event.user_id,
                'tags': event.tags,
                'correlation_id': event.correlation_id
            }
            
            # Sign payload
            headers = webhook.headers.copy()
            headers['Content-Type'] = 'application/json'
            
            if webhook.secret:
                signature = self._sign_payload(json.dumps(payload), webhook.secret)
                headers['X-Webhook-Signature'] = signature
            
            # Make request
            response = requests.post(
                webhook.url,
                json=payload,
                headers=headers,
                timeout=webhook.timeout_seconds
            )
            
            # Check response
            success = 200 <= response.status_code < 300
            
            return {
                'success': success,
                'status_code': response.status_code,
                'response_body': response.text,
                'error': None if success else f"HTTP {response.status_code}"
            }
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timeout'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Connection failed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _deliver_alternative(self, webhook: WebhookEndpoint, event: Event) -> Dict[str, Any]:
        """Deliver event via alternative method.
        
        Args:
            webhook: Webhook endpoint
            event: Event to deliver
            
        Returns:
            Delivery result
        """
        # Placeholder for alternative delivery methods (email, SMS, etc.)
        logger.info(f"Alternative delivery ({webhook.delivery_method.value}) not implemented")
        return {'success': True, 'message': 'Alternative delivery simulated'}
    
    def _schedule_retry(self, webhook: WebhookEndpoint, event: Event, attempt_number: int):
        """Schedule retry delivery.
        
        Args:
            webhook: Webhook endpoint
            event: Event to retry
            attempt_number: Retry attempt number
        """
        try:
            # Calculate retry delay (exponential backoff)
            delay_seconds = min(60, 2 ** (attempt_number - 2))  # 1, 2, 4, 8, 16, 32, 60...
            
            retry_time = datetime.now() + timedelta(seconds=delay_seconds)
            
            retry_item = {
                'webhook': webhook,
                'event': event,
                'attempt_number': attempt_number,
                'scheduled_time': retry_time
            }
            
            with self.lock:
                self.retry_queue[webhook.webhook_id].append(retry_item)
            
            logger.debug(f"Retry scheduled for {retry_time}: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Retry scheduling failed: {e}")
    
    def _process_retry_queue(self):
        """Process retry queue for failed deliveries."""
        try:
            current_time = datetime.now()
            
            with self.lock:
                for webhook_id, retries in list(self.retry_queue.items()):
                    ready_retries = []
                    remaining_retries = []
                    
                    for retry_item in retries:
                        if current_time >= retry_item['scheduled_time']:
                            ready_retries.append(retry_item)
                        else:
                            remaining_retries.append(retry_item)
                    
                    # Update retry queue
                    if remaining_retries:
                        self.retry_queue[webhook_id] = remaining_retries
                    else:
                        del self.retry_queue[webhook_id]
                    
                    # Process ready retries
                    for retry_item in ready_retries:
                        self._schedule_delivery(
                            retry_item['webhook'],
                            retry_item['event'],
                            retry_item['attempt_number']
                        )
                        
        except Exception as e:
            logger.error(f"Retry queue processing failed: {e}")
    
    def _sign_payload(self, payload: str, secret: str) -> str:
        """Sign webhook payload.
        
        Args:
            payload: JSON payload string
            secret: Webhook secret
            
        Returns:
            HMAC signature
        """
        try:
            signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return f"sha256={signature}"
            
        except Exception as e:
            logger.error(f"Payload signing failed: {e}")
            return ""
    
    def _generate_webhook_secret(self) -> str:
        """Generate secure webhook secret.
        
        Returns:
            Random secret string
        """
        import secrets
        return secrets.token_urlsafe(32)


class EventSystem:
    """High-level event system interface."""
    
    def __init__(self, webhook_manager: Optional[WebhookManager] = None):
        """Initialize event system.
        
        Args:
            webhook_manager: Optional webhook manager instance
        """
        self.webhook_manager = webhook_manager or WebhookManager()
        
        # Event handlers
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Event history
        self.event_history: deque = deque(maxlen=1000)
        
        logger.info("Event System initialized")
    
    def emit(self, event_type: EventType, data: Dict[str, Any],
            source: str = "system", **kwargs) -> str:
        """Emit event.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            **kwargs: Additional event properties
            
        Returns:
            Event ID
        """
        try:
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(),
                source=source,
                data=data,
                metadata=kwargs.get('metadata', {}),
                priority=EventPriority(kwargs.get('priority', 'normal')),
                tenant_id=kwargs.get('tenant_id'),
                user_id=kwargs.get('user_id'),
                tags=kwargs.get('tags', []),
                correlation_id=kwargs.get('correlation_id')
            )
            
            # Store in history
            self.event_history.append(event)
            
            # Call internal handlers
            self._call_internal_handlers(event)
            
            # Publish to webhooks
            self.webhook_manager.publish_event(event)
            
            logger.debug(f"Event emitted: {event_type.value} ({event.event_id})")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Event emission failed: {e}")
            raise
    
    def on(self, event_type: EventType, handler: Callable[[Event], None]):
        """Register event handler.
        
        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Event handler registered for {event_type.value}")
    
    def off(self, event_type: EventType, handler: Callable[[Event], None]):
        """Unregister event handler.
        
        Args:
            event_type: Event type
            handler: Handler function to remove
        """
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            logger.debug(f"Event handler unregistered for {event_type.value}")
    
    def get_recent_events(self, event_type: Optional[EventType] = None,
                         limit: int = 50) -> List[Event]:
        """Get recent events.
        
        Args:
            event_type: Optional event type filter
            limit: Maximum number of events
            
        Returns:
            List of recent events
        """
        events = list(self.event_history)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit]
    
    def create_webhook(self, name: str, url: str, event_types: List[str],
                      **kwargs) -> str:
        """Create webhook subscription.
        
        Args:
            name: Webhook name
            url: Webhook URL
            event_types: List of event types to subscribe to
            **kwargs: Additional webhook options
            
        Returns:
            Webhook ID
        """
        webhook_data = {
            'name': name,
            'url': url,
            'event_types': event_types,
            **kwargs
        }
        
        webhook_id = self.webhook_manager.register_webhook(webhook_data)
        
        # Create subscription
        self.webhook_manager.subscribe_to_events(
            webhook_id,
            event_types,
            kwargs.get('filters')
        )
        
        return webhook_id
    
    def remove_webhook(self, webhook_id: str) -> bool:
        """Remove webhook subscription.
        
        Args:
            webhook_id: Webhook identifier
            
        Returns:
            True if removal successful
        """
        return self.webhook_manager.unregister_webhook(webhook_id)
    
    def _call_internal_handlers(self, event: Event):
        """Call internal event handlers.
        
        Args:
            event: Event to handle
        """
        try:
            handlers = self.event_handlers.get(event.event_type, [])
            
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
                    
        except Exception as e:
            logger.error(f"Internal handler processing failed: {e}")
    
    def shutdown(self):
        """Shutdown event system."""
        logger.info("Shutting down event system...")
        
        if self.webhook_manager:
            self.webhook_manager.shutdown()
        
        logger.info("Event system shutdown complete")