"""Tests for email service module."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import pytest

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.services.email_service import (
    EmailConfig,
    EmailService,
    EmailTemplate,
    get_email_service,
)
from pynomaly.shared.exceptions import ConfigurationError, EmailError


class TestEmailConfig:
    """Test EmailConfig class."""

    def test_email_config_initialization(self):
        """Test EmailConfig initialization."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            smtp_username="test@gmail.com",
            smtp_password="password",
            smtp_use_tls=True,
            sender_email="test@gmail.com",
            sender_name="Test Sender",
            base_url="https://example.com"
        )

        assert config.smtp_server == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.smtp_username == "test@gmail.com"
        assert config.smtp_password == "password"
        assert config.smtp_use_tls is True
        assert config.sender_email == "test@gmail.com"
        assert config.sender_name == "Test Sender"
        assert config.base_url == "https://example.com"

    def test_email_config_validation(self):
        """Test EmailConfig validation."""
        # Test invalid SMTP port
        with pytest.raises(ValueError, match="SMTP port must be between 1 and 65535"):
            EmailConfig(smtp_port=0)

        with pytest.raises(ValueError, match="SMTP port must be between 1 and 65535"):
            EmailConfig(smtp_port=65536)

        # Test invalid email format
        with pytest.raises(ValueError, match="Invalid email format"):
            EmailConfig(sender_email="invalid_email")

        # Test missing required fields
        with pytest.raises(ValueError, match="SMTP server is required"):
            EmailConfig(smtp_server="")

        with pytest.raises(ValueError, match="SMTP username is required"):
            EmailConfig(smtp_username="")

        with pytest.raises(ValueError, match="SMTP password is required"):
            EmailConfig(smtp_password="")

        with pytest.raises(ValueError, match="Sender email is required"):
            EmailConfig(sender_email="")

    def test_email_config_is_configured(self):
        """Test is_configured method."""
        # Complete configuration
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )
        assert config.is_configured() is True

        # Missing required field
        incomplete_config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="",  # Missing password
            sender_email="test@gmail.com"
        )
        assert incomplete_config.is_configured() is False

    def test_email_config_from_settings(self):
        """Test creating EmailConfig from Settings."""
        settings = Settings()
        settings.smtp_server = "smtp.gmail.com"
        settings.smtp_port = 587
        settings.smtp_username = "test@gmail.com"
        settings.smtp_password = "password"
        settings.sender_email = "test@gmail.com"
        settings.sender_name = "Test Sender"
        settings.base_url = "https://example.com"

        config = EmailConfig.from_settings(settings)

        assert config.smtp_server == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.smtp_username == "test@gmail.com"
        assert config.smtp_password == "password"
        assert config.sender_email == "test@gmail.com"
        assert config.sender_name == "Test Sender"
        assert config.base_url == "https://example.com"

    def test_email_config_to_dict(self):
        """Test EmailConfig to_dict method."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        result = config.to_dict()

        assert result["smtp_server"] == "smtp.gmail.com"
        assert result["smtp_port"] == 587
        assert result["smtp_username"] == "test@gmail.com"
        assert result["smtp_password"] == "password"
        assert result["sender_email"] == "test@gmail.com"


class TestEmailTemplate:
    """Test EmailTemplate class."""

    def test_email_template_initialization(self):
        """Test EmailTemplate initialization."""
        template = EmailTemplate(
            subject="Test Subject: {name}",
            body="Hello {name}, this is a test email.",
            html_body="<h1>Hello {name}</h1><p>This is a test email.</p>",
            template_type="password_reset"
        )

        assert template.subject == "Test Subject: {name}"
        assert template.body == "Hello {name}, this is a test email."
        assert template.html_body == "<h1>Hello {name}</h1><p>This is a test email.</p>"
        assert template.template_type == "password_reset"

    def test_email_template_render(self):
        """Test template rendering."""
        template = EmailTemplate(
            subject="Test Subject: {name}",
            body="Hello {name}, your token is {token}.",
            html_body="<h1>Hello {name}</h1><p>Your token is <strong>{token}</strong>.</p>"
        )

        context = {"name": "John", "token": "abc123"}
        rendered = template.render(context)

        assert rendered["subject"] == "Test Subject: John"
        assert rendered["body"] == "Hello John, your token is abc123."
        assert rendered["html_body"] == "<h1>Hello John</h1><p>Your token is <strong>abc123</strong>.</p>"

    def test_email_template_render_missing_variable(self):
        """Test template rendering with missing variable."""
        template = EmailTemplate(
            subject="Test Subject: {name}",
            body="Hello {name}, your token is {token}."
        )

        context = {"name": "John"}  # Missing token
        rendered = template.render(context)

        assert rendered["subject"] == "Test Subject: John"
        assert rendered["body"] == "Hello John, your token is {token}."  # Should leave placeholder

    def test_email_template_validate(self):
        """Test template validation."""
        # Valid template
        template = EmailTemplate(
            subject="Test Subject",
            body="Test body"
        )

        errors = template.validate()
        assert len(errors) == 0

        # Invalid template
        invalid_template = EmailTemplate(
            subject="",  # Empty subject
            body="Test body"
        )

        errors = invalid_template.validate()
        assert len(errors) > 0
        assert "Subject cannot be empty" in errors

    def test_email_template_get_variables(self):
        """Test extracting variables from template."""
        template = EmailTemplate(
            subject="Hello {name}",
            body="Your {item} is ready. Code: {code}",
            html_body="<h1>Hello {name}</h1><p>Your {item} is ready.</p>"
        )

        variables = template.get_variables()

        assert "name" in variables
        assert "item" in variables
        assert "code" in variables
        assert len(variables) == 3

    def test_email_template_to_dict(self):
        """Test EmailTemplate to_dict method."""
        template = EmailTemplate(
            subject="Test Subject",
            body="Test body",
            html_body="<h1>Test</h1>",
            template_type="test"
        )

        result = template.to_dict()

        assert result["subject"] == "Test Subject"
        assert result["body"] == "Test body"
        assert result["html_body"] == "<h1>Test</h1>"
        assert result["template_type"] == "test"

    def test_email_template_from_dict(self):
        """Test EmailTemplate from_dict method."""
        data = {
            "subject": "Test Subject",
            "body": "Test body",
            "html_body": "<h1>Test</h1>",
            "template_type": "test"
        }

        template = EmailTemplate.from_dict(data)

        assert template.subject == "Test Subject"
        assert template.body == "Test body"
        assert template.html_body == "<h1>Test</h1>"
        assert template.template_type == "test"


class TestEmailService:
    """Test EmailService class."""

    def test_email_service_initialization(self):
        """Test EmailService initialization."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        assert service.config == config
        assert service.provider == EmailProvider.SMTP
        assert service.templates == {}
        assert service.connection is None

    def test_email_service_invalid_config(self):
        """Test EmailService with invalid configuration."""
        with pytest.raises(ConfigurationError):
            EmailService(None)

    def test_email_service_add_template(self):
        """Test adding email template."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        template = EmailTemplate(
            subject="Test Subject",
            body="Test body",
            template_type="test"
        )

        service.add_template("test", template)

        assert "test" in service.templates
        assert service.templates["test"] == template

    def test_email_service_get_template(self):
        """Test getting email template."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        template = EmailTemplate(
            subject="Test Subject",
            body="Test body",
            template_type="test"
        )

        service.add_template("test", template)

        retrieved = service.get_template("test")
        assert retrieved == template

        # Test non-existent template
        assert service.get_template("nonexistent") is None

    def test_email_service_create_message(self):
        """Test creating email message."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com",
            sender_name="Test Sender"
        )

        service = EmailService(config)

        message = service._create_message(
            to_email="recipient@example.com",
            subject="Test Subject",
            body="Test body",
            html_body="<h1>Test</h1>"
        )

        assert isinstance(message, MIMEMultipart)
        assert message["From"] == "Test Sender <test@gmail.com>"
        assert message["To"] == "recipient@example.com"
        assert message["Subject"] == "Test Subject"

    def test_email_service_create_message_plain_text(self):
        """Test creating plain text email message."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        message = service._create_message(
            to_email="recipient@example.com",
            subject="Test Subject",
            body="Test body"
        )

        assert isinstance(message, MIMEText)
        assert message["From"] == "test@gmail.com"
        assert message["To"] == "recipient@example.com"
        assert message["Subject"] == "Test Subject"

    @pytest.mark.asyncio
    async def test_email_service_send_email_success(self):
        """Test successful email sending."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}

            result = await service.send_email(
                to_email="recipient@example.com",
                subject="Test Subject",
                body="Test body"
            )

            assert result is True
            mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@gmail.com", "password")
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_service_send_email_failure(self):
        """Test email sending failure."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = smtplib.SMTPException("SMTP error")

            result = await service.send_email(
                to_email="recipient@example.com",
                subject="Test Subject",
                body="Test body"
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_email_service_send_email_with_template(self):
        """Test sending email with template."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        # Add template
        template = EmailTemplate(
            subject="Hello {name}",
            body="Welcome {name}!",
            html_body="<h1>Welcome {name}!</h1>"
        )
        service.add_template("welcome", template)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}

            result = await service.send_email_with_template(
                to_email="recipient@example.com",
                template_name="welcome",
                context={"name": "John"}
            )

            assert result is True
            mock_server.send_message.assert_called_once()

            # Check that template was rendered
            call_args = mock_server.send_message.call_args
            message = call_args[0][0]
            assert message["Subject"] == "Hello John"

    @pytest.mark.asyncio
    async def test_email_service_send_email_with_nonexistent_template(self):
        """Test sending email with non-existent template."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        with pytest.raises(EmailError, match="Template 'nonexistent' not found"):
            await service.send_email_with_template(
                to_email="recipient@example.com",
                template_name="nonexistent",
                context={}
            )

    @pytest.mark.asyncio
    async def test_email_service_send_password_reset_email(self):
        """Test sending password reset email."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com",
            base_url="https://example.com"
        )

        service = EmailService(config)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}

            result = await service.send_password_reset_email(
                email="user@example.com",
                reset_token="abc123",
                user_name="John Doe"
            )

            assert result is True
            mock_server.send_message.assert_called_once()

            # Check that message contains expected content
            call_args = mock_server.send_message.call_args
            message = call_args[0][0]
            assert "Password Reset" in message["Subject"]
            assert message["To"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_email_service_send_user_invitation_email(self):
        """Test sending user invitation email."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com",
            base_url="https://example.com"
        )

        service = EmailService(config)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}

            result = await service.send_user_invitation_email(
                email="newuser@example.com",
                invitation_token="xyz789",
                inviter_name="Admin User",
                organization_name="Test Org"
            )

            assert result is True
            mock_server.send_message.assert_called_once()

            # Check that message contains expected content
            call_args = mock_server.send_message.call_args
            message = call_args[0][0]
            assert "Invitation" in message["Subject"]
            assert message["To"] == "newuser@example.com"

    @pytest.mark.asyncio
    async def test_email_service_send_system_notification_email(self):
        """Test sending system notification email."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}

            result = await service.send_system_notification_email(
                email="admin@example.com",
                subject="System Alert",
                message="Something important happened",
                priority="high"
            )

            assert result is True
            mock_server.send_message.assert_called_once()

            # Check that message contains expected content
            call_args = mock_server.send_message.call_args
            message = call_args[0][0]
            assert message["Subject"] == "System Alert"
            assert message["To"] == "admin@example.com"

    @pytest.mark.asyncio
    async def test_email_service_send_bulk_emails(self):
        """Test sending bulk emails."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        recipients = [
            {"email": "user1@example.com", "name": "User 1"},
            {"email": "user2@example.com", "name": "User 2"},
            {"email": "user3@example.com", "name": "User 3"}
        ]

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}

            results = await service.send_bulk_emails(
                recipients=recipients,
                subject="Bulk Email Test",
                body="This is a bulk email test"
            )

            assert len(results) == 3
            assert all(results.values())  # All should be True
            assert mock_server.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_email_service_send_bulk_emails_with_template(self):
        """Test sending bulk emails with template."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        # Add template
        template = EmailTemplate(
            subject="Hello {name}",
            body="Welcome {name}!",
            html_body="<h1>Welcome {name}!</h1>"
        )
        service.add_template("welcome", template)

        recipients = [
            {"email": "user1@example.com", "name": "User 1"},
            {"email": "user2@example.com", "name": "User 2"}
        ]

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None
            mock_server.send_message.return_value = {}

            results = await service.send_bulk_emails_with_template(
                recipients=recipients,
                template_name="welcome"
            )

            assert len(results) == 2
            assert all(results.values())  # All should be True
            assert mock_server.send_message.call_count == 2

    def test_email_service_validate_email(self):
        """Test email validation."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        # Valid emails
        assert service.validate_email("test@example.com") is True
        assert service.validate_email("user.name@domain.co.uk") is True
        assert service.validate_email("test+tag@example.com") is True

        # Invalid emails
        assert service.validate_email("invalid") is False
        assert service.validate_email("invalid@") is False
        assert service.validate_email("@invalid.com") is False
        assert service.validate_email("test@") is False
        assert service.validate_email("") is False

    def test_email_service_get_connection_info(self):
        """Test getting connection info."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        info = service.get_connection_info()

        assert info["provider"] == "SMTP"
        assert info["server"] == "smtp.gmail.com"
        assert info["port"] == 587
        assert info["username"] == "test@gmail.com"
        assert info["use_tls"] is True
        assert "password" not in info  # Should not expose password

    def test_email_service_get_template_list(self):
        """Test getting template list."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        # Add templates
        template1 = EmailTemplate(subject="Test 1", body="Body 1", template_type="test1")
        template2 = EmailTemplate(subject="Test 2", body="Body 2", template_type="test2")

        service.add_template("template1", template1)
        service.add_template("template2", template2)

        templates = service.get_template_list()

        assert len(templates) == 2
        assert "template1" in templates
        assert "template2" in templates
        assert templates["template1"]["subject"] == "Test 1"
        assert templates["template2"]["subject"] == "Test 2"

    @pytest.mark.asyncio
    async def test_email_service_test_connection(self):
        """Test connection testing."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            mock_server.starttls.return_value = None
            mock_server.login.return_value = None

            result = await service.test_connection()

            assert result is True
            mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@gmail.com", "password")
            mock_server.quit.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_service_test_connection_failure(self):
        """Test connection testing failure."""
        config = EmailConfig(
            smtp_server="smtp.gmail.com",
            smtp_username="test@gmail.com",
            smtp_password="password",
            sender_email="test@gmail.com"
        )

        service = EmailService(config)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = smtplib.SMTPException("Connection failed")

            result = await service.test_connection()

            assert result is False


class TestEmailServiceHelpers:
    """Test email service helper functions."""

    def test_get_email_service_configured(self):
        """Test getting email service when configured."""
        with patch("pynomaly.infrastructure.services.email_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                smtp_server="smtp.gmail.com",
                smtp_username="test@gmail.com",
                smtp_password="password",
                sender_email="test@gmail.com"
            )

            service = get_email_service()

            assert service is not None
            assert isinstance(service, EmailService)

    def test_get_email_service_not_configured(self):
        """Test getting email service when not configured."""
        with patch("pynomaly.infrastructure.services.email_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                smtp_server=None,
                smtp_username=None,
                smtp_password=None,
                sender_email=None
            )

            service = get_email_service()

            assert service is None

    def test_get_email_service_singleton(self):
        """Test that email service is singleton."""
        with patch("pynomaly.infrastructure.services.email_service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                smtp_server="smtp.gmail.com",
                smtp_username="test@gmail.com",
                smtp_password="password",
                sender_email="test@gmail.com"
            )

            service1 = get_email_service()
            service2 = get_email_service()

            assert service1 is service2


class TestEmailProvider:
    """Test EmailProvider enum."""

    def test_email_provider_values(self):
        """Test EmailProvider enum values."""
        assert EmailProvider.SMTP == "smtp"
        assert EmailProvider.SENDGRID == "sendgrid"
        assert EmailProvider.MAILGUN == "mailgun"
        assert EmailProvider.SES == "ses"

    def test_email_provider_from_string(self):
        """Test creating EmailProvider from string."""
        assert EmailProvider("smtp") == EmailProvider.SMTP
        assert EmailProvider("sendgrid") == EmailProvider.SENDGRID
        assert EmailProvider("mailgun") == EmailProvider.MAILGUN
        assert EmailProvider("ses") == EmailProvider.SES
