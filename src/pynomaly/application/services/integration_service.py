"""
Integration service for managing third-party integrations and notifications.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from cryptography.fernet import Fernet

from pynomaly.domain.entities.integrations import (
    Integration,
    IntegrationConfig,
    IntegrationCredentials,
    IntegrationMetrics,
    IntegrationStatus,
    IntegrationType,
    NotificationHistory,
    NotificationLevel,
    NotificationPayload,
    NotificationTemplate,
)
from pynomaly.shared.exceptions import (
    AuthenticationError,
    IntegrationError,
    NotificationError,
    ValidationError,
)
from pynomaly.shared.types import TenantId, UserId


class IntegrationService:
    """Service for managing integrations and sending notifications."""

    def __init__(
        self, integration_repository, notification_repository, encryption_key: str
    ):
        self._integration_repo = integration_repository
        self._notification_repo = notification_repository
        self._fernet = Fernet(encryption_key.encode())
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    # Integration Management
    async def create_integration(
        self,
        name: str,
        integration_type: IntegrationType,
        tenant_id: TenantId,
        user_id: UserId,
        config: IntegrationConfig,
        credentials: dict[str, Any] | None = None,
    ) -> Integration:
        """Create a new integration."""
        # Validate configuration
        await self._validate_integration_config(integration_type, config)

        # Encrypt credentials if provided
        encrypted_credentials = None
        if credentials:
            encrypted_credentials = self._encrypt_credentials(credentials)

        # Create integration
        integration = Integration(
            id=str(uuid.uuid4()),
            name=name,
            integration_type=integration_type,
            tenant_id=tenant_id,
            created_by=user_id,
            status=(
                IntegrationStatus.PENDING_AUTH
                if credentials
                else IntegrationStatus.INACTIVE
            ),
            config=config,
            credentials=encrypted_credentials,
        )

        # Test connection if credentials provided
        if credentials:
            try:
                await self._test_integration_connection(integration, credentials)
                integration.status = IntegrationStatus.ACTIVE
            except Exception as e:
                integration.status = IntegrationStatus.ERROR
                integration.last_error = str(e)

        # Save integration
        return await self._integration_repo.create_integration(integration)

    async def update_integration_credentials(
        self, integration_id: str, user_id: UserId, credentials: dict[str, Any]
    ) -> Integration:
        """Update integration credentials."""
        integration = await self._integration_repo.get_integration_by_id(integration_id)
        if not integration:
            raise IntegrationError("Integration not found")

        # Check permissions
        if not (
            integration.created_by == user_id
            or await self._has_integration_permission(user_id, integration.tenant_id)
        ):
            raise AuthenticationError("Insufficient permissions")

        # Test connection with new credentials
        try:
            await self._test_integration_connection(integration, credentials)

            # Update credentials
            integration.credentials = self._encrypt_credentials(credentials)
            integration.status = IntegrationStatus.ACTIVE
            integration.last_error = None
            integration.updated_at = datetime.utcnow()

        except Exception as e:
            integration.status = IntegrationStatus.ERROR
            integration.last_error = str(e)

        return await self._integration_repo.update_integration(integration)

    async def get_integrations_for_tenant(
        self, tenant_id: TenantId
    ) -> list[Integration]:
        """Get all integrations for a tenant."""
        return await self._integration_repo.get_integrations_by_tenant(tenant_id)

    async def delete_integration(self, integration_id: str, user_id: UserId) -> bool:
        """Delete an integration."""
        integration = await self._integration_repo.get_integration_by_id(integration_id)
        if not integration:
            return False

        # Check permissions
        if not (
            integration.created_by == user_id
            or await self._has_integration_permission(user_id, integration.tenant_id)
        ):
            raise AuthenticationError("Insufficient permissions")

        return await self._integration_repo.delete_integration(integration_id)

    # Notification Sending
    async def send_notification(
        self,
        tenant_id: TenantId,
        payload: NotificationPayload,
        integration_types: list[IntegrationType] | None = None,
    ) -> dict[str, bool]:
        """Send notification to all matching integrations."""
        # Get active integrations for tenant
        integrations = await self._integration_repo.get_active_integrations_by_tenant(
            tenant_id
        )

        # Filter by integration types if specified
        if integration_types:
            integrations = [
                i for i in integrations if i.integration_type in integration_types
            ]

        # Filter by triggers and notification levels
        filtered_integrations = []
        for integration in integrations:
            if (
                payload.trigger_type in integration.config.triggers
                and payload.level in integration.config.notification_levels
            ):
                filtered_integrations.append(integration)

        # Send notifications concurrently
        tasks = []
        for integration in filtered_integrations:
            task = asyncio.create_task(
                self._send_notification_to_integration(integration, payload)
            )
            tasks.append((integration.id, task))

        # Wait for all notifications to complete
        results = {}
        for integration_id, task in tasks:
            try:
                success = await task
                results[integration_id] = success
            except Exception as e:
                results[integration_id] = False
                # Log error
                print(
                    f"Failed to send notification to integration {integration_id}: {e}"
                )

        return results

    async def _send_notification_to_integration(
        self, integration: Integration, payload: NotificationPayload
    ) -> bool:
        """Send notification to a specific integration."""
        start_time = datetime.utcnow()
        history_record = NotificationHistory(
            id=str(uuid.uuid4()),
            integration_id=integration.id,
            payload=payload,
            response_status=0,
            response_body="",
        )

        try:
            # Apply template if configured
            rendered_payload = await self._apply_template(integration, payload)

            # Apply filters
            if not await self._passes_filters(integration, rendered_payload):
                return True  # Filtered out, but not an error

            # Send notification based on integration type
            if integration.integration_type == IntegrationType.SLACK:
                success = await self._send_slack_notification(
                    integration, rendered_payload, history_record
                )
            elif integration.integration_type == IntegrationType.PAGERDUTY:
                success = await self._send_pagerduty_notification(
                    integration, rendered_payload, history_record
                )
            elif integration.integration_type == IntegrationType.TEAMS:
                success = await self._send_teams_notification(
                    integration, rendered_payload, history_record
                )
            elif integration.integration_type == IntegrationType.WEBHOOK:
                success = await self._send_webhook_notification(
                    integration, rendered_payload, history_record
                )
            elif integration.integration_type == IntegrationType.EMAIL:
                success = await self._send_email_notification(
                    integration, rendered_payload, history_record
                )
            else:
                raise NotificationError(
                    f"Unsupported integration type: {integration.integration_type}"
                )

            # Update integration metrics
            await self._update_integration_metrics(integration.id, success)

            # Calculate delivery time
            delivery_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            history_record.delivery_time_ms = int(delivery_time)

            return success

        except Exception as e:
            history_record.error_message = str(e)
            history_record.response_status = 500
            await self._update_integration_metrics(integration.id, False)
            raise

        finally:
            # Save notification history
            await self._notification_repo.create_notification_history(history_record)

    async def _send_slack_notification(
        self,
        integration: Integration,
        payload: NotificationPayload,
        history_record: NotificationHistory,
    ) -> bool:
        """Send notification to Slack."""
        if not integration.credentials:
            raise NotificationError("Slack credentials not configured")

        credentials = self._decrypt_credentials(integration.credentials)
        webhook_url = credentials.get("webhook_url")

        if not webhook_url:
            raise NotificationError("Slack webhook URL not configured")

        # Format message for Slack
        slack_message = payload.to_slack_format()

        # Add custom fields from integration config
        if integration.config.settings.get("include_tenant_info"):
            slack_message["attachments"][0]["fields"].append(
                {"title": "Tenant", "value": str(payload.tenant_id), "short": True}
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=slack_message,
                timeout=aiohttp.ClientTimeout(total=integration.config.timeout_seconds),
            ) as response:
                history_record.response_status = response.status
                history_record.response_body = await response.text()

                return response.status == 200

    async def _send_pagerduty_notification(
        self,
        integration: Integration,
        payload: NotificationPayload,
        history_record: NotificationHistory,
    ) -> bool:
        """Send notification to PagerDuty."""
        if not integration.credentials:
            raise NotificationError("PagerDuty credentials not configured")

        credentials = self._decrypt_credentials(integration.credentials)
        routing_key = credentials.get("routing_key")

        if not routing_key:
            raise NotificationError("PagerDuty routing key not configured")

        # Format event for PagerDuty
        pagerduty_event = payload.to_pagerduty_format()
        pagerduty_event["routing_key"] = routing_key

        # Add dedup_key to prevent duplicate incidents
        pagerduty_event["dedup_key"] = (
            f"pynomaly-{payload.trigger_type.value}-{payload.timestamp.isoformat()}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=pagerduty_event,
                timeout=aiohttp.ClientTimeout(total=integration.config.timeout_seconds),
            ) as response:
                history_record.response_status = response.status
                history_record.response_body = await response.text()

                return response.status == 202

    async def _send_teams_notification(
        self,
        integration: Integration,
        payload: NotificationPayload,
        history_record: NotificationHistory,
    ) -> bool:
        """Send notification to Microsoft Teams."""
        if not integration.credentials:
            raise NotificationError("Teams credentials not configured")

        credentials = self._decrypt_credentials(integration.credentials)
        webhook_url = credentials.get("webhook_url")

        if not webhook_url:
            raise NotificationError("Teams webhook URL not configured")

        # Format message for Teams
        teams_message = payload.to_teams_format()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=teams_message,
                timeout=aiohttp.ClientTimeout(total=integration.config.timeout_seconds),
            ) as response:
                history_record.response_status = response.status
                history_record.response_body = await response.text()

                return response.status == 200

    async def _send_webhook_notification(
        self,
        integration: Integration,
        payload: NotificationPayload,
        history_record: NotificationHistory,
    ) -> bool:
        """Send notification to generic webhook."""
        if not integration.credentials:
            raise NotificationError("Webhook credentials not configured")

        credentials = self._decrypt_credentials(integration.credentials)
        webhook_url = credentials.get("url")

        if not webhook_url:
            raise NotificationError("Webhook URL not configured")

        # Format message for webhook
        webhook_payload = payload.to_webhook_format()

        # Add authentication headers if configured
        headers = {"Content-Type": "application/json"}
        if credentials.get("auth_header"):
            headers["Authorization"] = credentials["auth_header"]

        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=webhook_payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=integration.config.timeout_seconds),
            ) as response:
                history_record.response_status = response.status
                history_record.response_body = await response.text()

                return 200 <= response.status < 300

    async def _send_email_notification(
        self,
        integration: Integration,
        payload: NotificationPayload,
        history_record: NotificationHistory,
    ) -> bool:
        """Send email notification."""
        # TODO: Implement email sending using SMTP or email service
        # For now, just simulate success
        history_record.response_status = 200
        history_record.response_body = "Email sent successfully"
        return True

    # Template Management
    async def create_notification_template(
        self, template: NotificationTemplate
    ) -> NotificationTemplate:
        """Create a new notification template."""
        template.id = str(uuid.uuid4())
        return await self._notification_repo.create_template(template)

    async def get_templates_for_integration(
        self, integration_type: IntegrationType, tenant_id: TenantId
    ) -> list[NotificationTemplate]:
        """Get all templates for an integration type and tenant."""
        return await self._notification_repo.get_templates(integration_type, tenant_id)

    async def _apply_template(
        self, integration: Integration, payload: NotificationPayload
    ) -> NotificationPayload:
        """Apply notification template to payload."""
        if (
            not integration.config.template_id
            and not integration.config.custom_template
        ):
            return payload  # No template configured

        template = None
        if integration.config.template_id:
            template = await self._notification_repo.get_template_by_id(
                integration.config.template_id
            )

        if not template and integration.config.custom_template:
            # Use custom template from config
            template = NotificationTemplate(
                id="custom",
                name="Custom Template",
                integration_type=integration.integration_type,
                trigger_types=[payload.trigger_type],
                title_template=integration.config.custom_template.get(
                    "title", payload.title
                ),
                message_template=integration.config.custom_template.get(
                    "message", payload.message
                ),
                tenant_id=integration.tenant_id,
                created_by=integration.created_by,
            )

        if template and payload.trigger_type in template.trigger_types:
            # Prepare template context
            context = {
                "title": payload.title,
                "message": payload.message,
                "level": payload.level.value,
                "trigger_type": payload.trigger_type.value,
                "timestamp": payload.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                **payload.data,
            }

            # Render template
            rendered = template.render(context)
            payload.title = rendered["title"]
            payload.message = rendered["message"]

        return payload

    async def _passes_filters(
        self, integration: Integration, payload: NotificationPayload
    ) -> bool:
        """Check if payload passes integration filters."""
        filters = integration.config.filters

        # Level filter
        if "min_level" in filters:
            level_order = {
                NotificationLevel.INFO: 0,
                NotificationLevel.WARNING: 1,
                NotificationLevel.ERROR: 2,
                NotificationLevel.CRITICAL: 3,
            }
            if level_order.get(payload.level, 0) < level_order.get(
                filters["min_level"], 0
            ):
                return False

        # Keyword filters
        if "keywords" in filters:
            keywords = filters["keywords"]
            if keywords and not any(
                keyword.lower() in payload.message.lower() for keyword in keywords
            ):
                return False

        # Time-based filters
        if "quiet_hours" in filters:
            quiet_hours = filters["quiet_hours"]
            current_hour = payload.timestamp.hour
            if quiet_hours["start"] <= current_hour <= quiet_hours["end"]:
                # Skip if it's during quiet hours and not critical
                if payload.level != NotificationLevel.CRITICAL:
                    return False

        return True

    # Utility Methods
    def _encrypt_credentials(
        self, credentials: dict[str, Any]
    ) -> IntegrationCredentials:
        """Encrypt credentials for secure storage."""
        credentials_json = json.dumps(credentials)
        encrypted_data = self._fernet.encrypt(credentials_json.encode()).decode()

        return IntegrationCredentials(
            encrypted_data=encrypted_data,
            encryption_key_id="default",  # Could use key rotation in the future
            expires_at=datetime.utcnow() + timedelta(days=365),  # 1 year expiry
        )

    def _decrypt_credentials(
        self, credentials: IntegrationCredentials
    ) -> dict[str, Any]:
        """Decrypt credentials for use."""
        try:
            decrypted_data = self._fernet.decrypt(
                credentials.encrypted_data.encode()
            ).decode()
            return json.loads(decrypted_data)
        except Exception as e:
            raise AuthenticationError(f"Failed to decrypt credentials: {e}")

    async def _test_integration_connection(
        self, integration: Integration, credentials: dict[str, Any]
    ) -> bool:
        """Test integration connection."""
        if integration.integration_type == IntegrationType.SLACK:
            webhook_url = credentials.get("webhook_url")
            if not webhook_url:
                raise ValidationError("Slack webhook URL required")

            # Test with a simple message
            test_message = {"text": "Pynomaly integration test"}
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=test_message) as response:
                    return response.status == 200

        elif integration.integration_type == IntegrationType.PAGERDUTY:
            routing_key = credentials.get("routing_key")
            if not routing_key:
                raise ValidationError("PagerDuty routing key required")

            # Test with a test event
            test_event = {
                "routing_key": routing_key,
                "event_action": "trigger",
                "payload": {
                    "summary": "Pynomaly integration test",
                    "source": "pynomaly-test",
                    "severity": "info",
                },
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue", json=test_event
                ) as response:
                    return response.status == 202

        # Add other integration tests as needed
        return True

    async def _validate_integration_config(
        self, integration_type: IntegrationType, config: IntegrationConfig
    ) -> None:
        """Validate integration configuration."""
        if not config.notification_levels:
            raise ValidationError("At least one notification level must be configured")

        if not config.triggers:
            raise ValidationError("At least one trigger type must be configured")

        if config.retry_count < 0 or config.retry_count > 10:
            raise ValidationError("Retry count must be between 0 and 10")

        if config.timeout_seconds < 5 or config.timeout_seconds > 300:
            raise ValidationError("Timeout must be between 5 and 300 seconds")

    async def _has_integration_permission(
        self, user_id: UserId, tenant_id: TenantId
    ) -> bool:
        """Check if user has permission to manage integrations for tenant."""
        # TODO: Implement proper permission checking
        return True

    async def _update_integration_metrics(
        self, integration_id: str, success: bool
    ) -> None:
        """Update integration performance metrics."""
        metrics = await self._integration_repo.get_integration_metrics(integration_id)
        if not metrics:
            metrics = IntegrationMetrics(integration_id=integration_id)

        metrics.total_notifications += 1
        if success:
            metrics.successful_notifications += 1
            metrics.last_success = datetime.utcnow()
        else:
            metrics.failed_notifications += 1
            metrics.last_failure = datetime.utcnow()

        await self._integration_repo.update_integration_metrics(metrics)

    # Public API Methods
    async def get_integration_metrics(
        self, integration_id: str
    ) -> IntegrationMetrics | None:
        """Get performance metrics for an integration."""
        return await self._integration_repo.get_integration_metrics(integration_id)

    async def get_notification_history(
        self, integration_id: str, limit: int = 100, offset: int = 0
    ) -> list[NotificationHistory]:
        """Get notification history for an integration."""
        return await self._notification_repo.get_notification_history(
            integration_id, limit, offset
        )

    async def retry_failed_notification(self, history_id: str) -> bool:
        """Retry a failed notification."""
        history = await self._notification_repo.get_notification_history_by_id(
            history_id
        )
        if not history:
            return False

        integration = await self._integration_repo.get_integration_by_id(
            history.integration_id
        )
        if not integration:
            return False

        # Create new history record for retry
        history.retry_count += 1
        return await self._send_notification_to_integration(
            integration, history.payload
        )
