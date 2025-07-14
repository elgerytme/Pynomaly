"""
Email service for sending report notifications and deliveries.

This service provides:
- Email delivery with attachments
- HTML email templates
- Bulk email sending
- Email tracking and delivery status
- Integration with various email providers
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import aiosmtplib
from jinja2 import Template

from ...shared.exceptions import NotificationError, ValidationError


@dataclass
class EmailConfig:
    """Configuration for email service."""

    # SMTP Configuration
    smtp_server: str
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False

    # Default sender information
    from_email: str = ""
    from_name: str = "Pynomaly System"
    reply_to: str | None = None

    # Email templates directory
    templates_dir: str | None = None

    # Delivery configuration
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    batch_size: int = 50
    rate_limit: int = 100  # emails per hour


@dataclass
class EmailMessage:
    """Email message configuration."""

    to: list[str]
    subject: str
    body: str
    html_body: str | None = None
    attachments: list[str] = None
    cc: list[str] = None
    bcc: list[str] = None
    reply_to: str | None = None
    priority: str = "normal"  # low, normal, high

    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.cc is None:
            self.cc = []
        if self.bcc is None:
            self.bcc = []


@dataclass
class EmailDeliveryResult:
    """Result of email delivery attempt."""

    message_id: str
    recipients: list[str]
    status: str  # sent, failed, retry
    timestamp: datetime
    error: str | None = None
    smtp_response: str | None = None
    delivery_time: float | None = None


class EmailService:
    """Service for sending emails with templates and attachments."""

    def __init__(self, config: EmailConfig):
        """Initialize email service with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._delivery_history: list[EmailDeliveryResult] = []
        self._rate_limiter = asyncio.Semaphore(config.rate_limit)

        # Load email templates
        self._templates = {}
        if config.templates_dir:
            self._load_templates()

    async def send_email(
        self,
        message: EmailMessage,
        template_name: str | None = None,
        template_data: dict[str, Any] | None = None,
    ) -> EmailDeliveryResult:
        """Send a single email message."""
        start_time = datetime.now()
        message_id = f"email_{int(start_time.timestamp())}"

        try:
            # Rate limiting
            async with self._rate_limiter:
                # Render template if specified
                if template_name:
                    await self._render_template(
                        message, template_name, template_data or {}
                    )

                # Validate message
                self._validate_message(message)

                # Send email
                smtp_response = await self._send_via_smtp(message)

                delivery_time = (datetime.now() - start_time).total_seconds()

                result = EmailDeliveryResult(
                    message_id=message_id,
                    recipients=message.to + message.cc + message.bcc,
                    status="sent",
                    timestamp=start_time,
                    smtp_response=smtp_response,
                    delivery_time=delivery_time,
                )

                self._delivery_history.append(result)
                self.logger.info(
                    f"Email sent successfully to {len(result.recipients)} recipients"
                )

                return result

        except Exception as e:
            result = EmailDeliveryResult(
                message_id=message_id,
                recipients=message.to + message.cc + message.bcc,
                status="failed",
                timestamp=start_time,
                error=str(e),
            )

            self._delivery_history.append(result)
            self.logger.error(f"Failed to send email: {str(e)}")

            raise NotificationError(f"Failed to send email: {str(e)}") from e

    async def send_bulk_emails(
        self,
        messages: list[EmailMessage],
        template_name: str | None = None,
        template_data: dict[str, Any] | None = None,
    ) -> list[EmailDeliveryResult]:
        """Send multiple emails in batches."""
        results = []

        # Process in batches
        for i in range(0, len(messages), self.config.batch_size):
            batch = messages[i : i + self.config.batch_size]

            # Send batch concurrently
            batch_tasks = [
                self.send_email(msg, template_name, template_data) for msg in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    # Create failed result for exception
                    failed_result = EmailDeliveryResult(
                        message_id=f"failed_{int(datetime.now().timestamp())}",
                        recipients=[],
                        status="failed",
                        timestamp=datetime.now(),
                        error=str(result),
                    )
                    results.append(failed_result)
                else:
                    results.append(result)

            # Rate limiting delay between batches
            if i + self.config.batch_size < len(messages):
                await asyncio.sleep(1)

        self.logger.info(
            f"Bulk email sending completed: {len(results)} emails processed"
        )
        return results

    async def send_report_email(
        self,
        recipients: list[str],
        subject: str,
        report_data: dict[str, Any],
        attachments: list[str] | None = None,
        template_name: str = "anomaly_report",
    ) -> EmailDeliveryResult:
        """Send a report email with anomaly detection results."""
        message = EmailMessage(
            to=recipients,
            subject=subject,
            body="",  # Will be filled by template
            attachments=attachments or [],
        )

        return await self.send_email(
            message=message, template_name=template_name, template_data=report_data
        )

    async def send_alert_email(
        self, recipients: list[str], alert_data: dict[str, Any], priority: str = "high"
    ) -> EmailDeliveryResult:
        """Send an alert email for high-priority anomalies."""
        subject = (
            f"🚨 Anomaly Alert: {alert_data.get('alert_type', 'High Priority Anomaly')}"
        )

        message = EmailMessage(
            to=recipients,
            subject=subject,
            body="",  # Will be filled by template
            priority=priority,
        )

        return await self.send_email(
            message=message, template_name="anomaly_alert", template_data=alert_data
        )

    async def send_schedule_notification(
        self,
        recipients: list[str],
        schedule_name: str,
        execution_result: dict[str, Any],
        notification_type: str = "success",
    ) -> EmailDeliveryResult:
        """Send notification about scheduled task execution."""
        if notification_type == "success":
            subject = f"✅ Schedule Completed: {schedule_name}"
            template_name = "schedule_success"
        else:
            subject = f"❌ Schedule Failed: {schedule_name}"
            template_name = "schedule_failure"

        message = EmailMessage(
            to=recipients,
            subject=subject,
            body="",  # Will be filled by template
        )

        template_data = {
            "schedule_name": schedule_name,
            "execution_result": execution_result,
            "notification_type": notification_type,
        }

        return await self.send_email(
            message=message, template_name=template_name, template_data=template_data
        )

    def _validate_message(self, message: EmailMessage) -> None:
        """Validate email message."""
        if not message.to:
            raise ValidationError("Email message must have at least one recipient")

        if not message.subject:
            raise ValidationError("Email message must have a subject")

        if not message.body and not message.html_body:
            raise ValidationError("Email message must have body content")

        # Validate email addresses
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        for email in message.to + message.cc + message.bcc:
            if not re.match(email_pattern, email):
                raise ValidationError(f"Invalid email address: {email}")

    async def _send_via_smtp(self, message: EmailMessage) -> str:
        """Send email via SMTP."""
        # Create MIME message
        mime_msg = MIMEMultipart("alternative")
        mime_msg["From"] = f"{self.config.from_name} <{self.config.from_email}>"
        mime_msg["To"] = ", ".join(message.to)
        mime_msg["Subject"] = message.subject

        if message.cc:
            mime_msg["Cc"] = ", ".join(message.cc)

        if message.reply_to or self.config.reply_to:
            mime_msg["Reply-To"] = message.reply_to or self.config.reply_to

        # Set priority
        if message.priority == "high":
            mime_msg["X-Priority"] = "1"
            mime_msg["X-MSMail-Priority"] = "High"
        elif message.priority == "low":
            mime_msg["X-Priority"] = "5"
            mime_msg["X-MSMail-Priority"] = "Low"

        # Add body content
        if message.body:
            text_part = MIMEText(message.body, "plain", "utf-8")
            mime_msg.attach(text_part)

        if message.html_body:
            html_part = MIMEText(message.html_body, "html", "utf-8")
            mime_msg.attach(html_part)

        # Add attachments
        for attachment_path in message.attachments:
            await self._add_attachment(mime_msg, attachment_path)

        # Send email
        all_recipients = message.to + message.cc + message.bcc

        if self.config.use_ssl:
            # Use SSL
            server = aiosmtplib.SMTP(
                hostname=self.config.smtp_server,
                port=self.config.smtp_port,
                use_tls=False,
                use_ssl=True,
            )
        else:
            # Use TLS
            server = aiosmtplib.SMTP(
                hostname=self.config.smtp_server,
                port=self.config.smtp_port,
                use_tls=self.config.use_tls,
            )

        try:
            await server.connect()

            if self.config.username and self.config.password:
                await server.login(self.config.username, self.config.password)

            response = await server.send_message(mime_msg, recipients=all_recipients)
            return str(response)

        finally:
            await server.quit()

    async def _add_attachment(self, mime_msg: MIMEMultipart, file_path: str) -> None:
        """Add file attachment to MIME message."""
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.warning(f"Attachment file not found: {file_path}")
                return

            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())

            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {path.name}")

            mime_msg.attach(part)

        except Exception as e:
            self.logger.error(f"Failed to add attachment {file_path}: {str(e)}")

    async def _render_template(
        self, message: EmailMessage, template_name: str, template_data: dict[str, Any]
    ) -> None:
        """Render email template with data."""
        try:
            if template_name not in self._templates:
                raise ValidationError(f"Email template not found: {template_name}")

            template_config = self._templates[template_name]

            # Render subject
            if "subject" in template_config:
                subject_template = Template(template_config["subject"])
                message.subject = subject_template.render(**template_data)

            # Render text body
            if "text" in template_config:
                text_template = Template(template_config["text"])
                message.body = text_template.render(**template_data)

            # Render HTML body
            if "html" in template_config:
                html_template = Template(template_config["html"])
                message.html_body = html_template.render(**template_data)

        except Exception as e:
            raise NotificationError(f"Failed to render email template: {str(e)}") from e

    def _load_templates(self) -> None:
        """Load email templates from directory."""
        try:
            templates_dir = Path(self.config.templates_dir)
            if not templates_dir.exists():
                templates_dir.mkdir(parents=True)
                self._create_default_templates()

            # Load template files
            for template_file in templates_dir.glob("*.json"):
                template_name = template_file.stem
                with open(template_file) as f:
                    import json

                    template_config = json.load(f)
                    self._templates[template_name] = template_config

            self.logger.info(f"Loaded {len(self._templates)} email templates")

        except Exception as e:
            self.logger.error(f"Failed to load email templates: {str(e)}")
            self._create_default_templates()

    def _create_default_templates(self) -> None:
        """Create default email templates."""
        default_templates = {
            "anomaly_report": {
                "subject": "Anomaly Detection Report - {{ timestamp }}",
                "text": """
Anomaly Detection Report

Generated: {{ timestamp }}
Total Records: {{ metrics.total_records }}
Anomalies Detected: {{ metrics.anomalies_detected }}
Anomaly Rate: {{ metrics.anomaly_rate }}%

{% if metrics.anomalies_detected > 0 %}
High Priority Anomalies:
{% for anomaly in high_priority_anomalies %}
- {{ anomaly.timestamp }}: Score {{ anomaly.anomaly_score }}
{% endfor %}
{% endif %}

This report was generated automatically by Pynomaly.
                """,
                "html": """
<html>
<body>
<h2>Anomaly Detection Report</h2>

<p><strong>Generated:</strong> {{ timestamp }}</p>

<table border="1" style="border-collapse: collapse;">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Records</td><td>{{ metrics.total_records }}</td></tr>
<tr><td>Anomalies Detected</td><td>{{ metrics.anomalies_detected }}</td></tr>
<tr><td>Anomaly Rate</td><td>{{ metrics.anomaly_rate }}%</td></tr>
</table>

{% if metrics.anomalies_detected > 0 %}
<h3>High Priority Anomalies</h3>
<ul>
{% for anomaly in high_priority_anomalies %}
<li>{{ anomaly.timestamp }}: Score {{ anomaly.anomaly_score }}</li>
{% endfor %}
</ul>
{% endif %}

<p><em>This report was generated automatically by Pynomaly.</em></p>
</body>
</html>
                """,
            },
            "anomaly_alert": {
                "subject": "🚨 Anomaly Alert: {{ alert_type }}",
                "text": """
ANOMALY ALERT

Alert Type: {{ alert_type }}
Timestamp: {{ timestamp }}
Anomaly Score: {{ anomaly_score }}
Confidence: {{ confidence }}

Details:
{{ details }}

Immediate action may be required.
                """,
                "html": """
<html>
<body style="font-family: Arial, sans-serif;">
<div style="background-color: #ff6b6b; color: white; padding: 10px;">
<h2>🚨 ANOMALY ALERT</h2>
</div>

<p><strong>Alert Type:</strong> {{ alert_type }}</p>
<p><strong>Timestamp:</strong> {{ timestamp }}</p>
<p><strong>Anomaly Score:</strong> <span style="color: red; font-weight: bold;">{{ anomaly_score }}</span></p>
<p><strong>Confidence:</strong> {{ confidence }}</p>

<h3>Details</h3>
<p>{{ details }}</p>

<div style="background-color: #ffe66d; padding: 10px; border-left: 5px solid #ff6b6b;">
<strong>⚠️ Immediate action may be required.</strong>
</div>
</body>
</html>
                """,
            },
            "schedule_success": {
                "subject": "✅ Schedule Completed: {{ schedule_name }}",
                "text": """
Schedule Execution Successful

Schedule: {{ schedule_name }}
Execution Time: {{ execution_result.start_time }}
Duration: {{ execution_result.duration }}
Status: {{ execution_result.status }}

Results:
{{ execution_result.summary }}
                """,
                "html": """
<html>
<body>
<div style="background-color: #4ecdc4; color: white; padding: 10px;">
<h2>✅ Schedule Execution Successful</h2>
</div>

<p><strong>Schedule:</strong> {{ schedule_name }}</p>
<p><strong>Execution Time:</strong> {{ execution_result.start_time }}</p>
<p><strong>Duration:</strong> {{ execution_result.duration }}</p>
<p><strong>Status:</strong> {{ execution_result.status }}</p>

<h3>Results</h3>
<p>{{ execution_result.summary }}</p>
</body>
</html>
                """,
            },
            "schedule_failure": {
                "subject": "❌ Schedule Failed: {{ schedule_name }}",
                "text": """
Schedule Execution Failed

Schedule: {{ schedule_name }}
Execution Time: {{ execution_result.start_time }}
Status: {{ execution_result.status }}
Error: {{ execution_result.error }}

Please review the schedule configuration and logs.
                """,
                "html": """
<html>
<body>
<div style="background-color: #ff6b6b; color: white; padding: 10px;">
<h2>❌ Schedule Execution Failed</h2>
</div>

<p><strong>Schedule:</strong> {{ schedule_name }}</p>
<p><strong>Execution Time:</strong> {{ execution_result.start_time }}</p>
<p><strong>Status:</strong> {{ execution_result.status }}</p>
<p><strong>Error:</strong> <span style="color: red;">{{ execution_result.error }}</span></p>

<p>Please review the schedule configuration and logs.</p>
</body>
</html>
                """,
            },
        }

        self._templates.update(default_templates)

    def get_delivery_stats(self) -> dict[str, Any]:
        """Get email delivery statistics."""
        total_sent = len([r for r in self._delivery_history if r.status == "sent"])
        total_failed = len([r for r in self._delivery_history if r.status == "failed"])

        if self._delivery_history:
            avg_delivery_time = sum(
                r.delivery_time
                for r in self._delivery_history
                if r.delivery_time is not None
            ) / len([r for r in self._delivery_history if r.delivery_time is not None])
        else:
            avg_delivery_time = 0

        return {
            "total_emails": len(self._delivery_history),
            "successful_deliveries": total_sent,
            "failed_deliveries": total_failed,
            "success_rate": (total_sent / len(self._delivery_history) * 100)
            if self._delivery_history
            else 0,
            "average_delivery_time": avg_delivery_time,
        }

    def get_delivery_history(self, limit: int = 100) -> list[EmailDeliveryResult]:
        """Get recent email delivery history."""
        return sorted(self._delivery_history, key=lambda r: r.timestamp, reverse=True)[
            :limit
        ]
