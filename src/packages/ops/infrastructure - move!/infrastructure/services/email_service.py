"""Email service for sending notifications and authentication emails.

This module provides a comprehensive email service for the Pynomaly platform,
supporting password reset, user invitations, and system notifications.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, EmailStr

from pynomaly.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class EmailConfig(BaseModel):
    """Email configuration settings."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    smtp_server: str
    smtp_port: int = 587
    smtp_username: str
    smtp_password: str
    use_tls: bool = True
    sender_email: EmailStr
    sender_name: str = "Pynomaly System"
    base_url: str = "http://localhost:8000"


class EmailTemplate(BaseModel):
    """Email template model."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    subject: str
    html_content: str
    text_content: str


class EmailService:
    """Email service for sending various types of emails."""

    def __init__(self, config: EmailConfig):
        """Initialize email service with configuration."""
        self.config = config

    async def send_password_reset_email(
        self, email: EmailStr, reset_token: str, user_name: str = "User"
    ) -> bool:
        """Send password reset email with reset link.

        Args:
            email: Recipient email address
            reset_token: Password reset token
            user_name: User's name for personalization

        Returns:
            True if email was sent successfully
        """
        try:
            # Create reset link
            reset_link = urljoin(
                self.config.base_url, f"/reset-password?token={reset_token}"
            )

            # Generate email content
            template = self._get_password_reset_template(
                user_name=user_name, reset_link=reset_link
            )

            return await self._send_email(
                to_email=email,
                subject=template.subject,
                html_content=template.html_content,
                text_content=template.text_content,
            )

        except Exception as e:
            logger.error(f"Failed to send password reset email to {email}: {e}")
            return False

    async def send_user_invitation_email(
        self,
        email: EmailStr,
        invitation_token: str,
        inviter_name: str = "Pynomaly Team",
        organization_name: str = "Pynomaly",
    ) -> bool:
        """Send user invitation email.

        Args:
            email: Recipient email address
            invitation_token: Invitation token
            inviter_name: Name of person sending invitation
            organization_name: Organization name

        Returns:
            True if email was sent successfully
        """
        try:
            # Create invitation link
            invitation_link = urljoin(
                self.config.base_url, f"/accept-invitation?token={invitation_token}"
            )

            # Generate email content
            template = self._get_user_invitation_template(
                inviter_name=inviter_name,
                organization_name=organization_name,
                invitation_link=invitation_link,
            )

            return await self._send_email(
                to_email=email,
                subject=template.subject,
                html_content=template.html_content,
                text_content=template.text_content,
            )

        except Exception as e:
            logger.error(f"Failed to send invitation email to {email}: {e}")
            return False

    async def send_system_notification_email(
        self, email: EmailStr, subject: str, message: str, priority: str = "normal"
    ) -> bool:
        """Send system notification email.

        Args:
            email: Recipient email address
            subject: Email subject
            message: Email message
            priority: Email priority (low, normal, high, urgent)

        Returns:
            True if email was sent successfully
        """
        try:
            # Generate email content
            template = self._get_system_notification_template(
                subject=subject, message=message, priority=priority
            )

            return await self._send_email(
                to_email=email,
                subject=template.subject,
                html_content=template.html_content,
                text_content=template.text_content,
            )

        except Exception as e:
            logger.error(f"Failed to send system notification to {email}: {e}")
            return False

    async def _send_email(
        self, to_email: EmailStr, subject: str, html_content: str, text_content: str
    ) -> bool:
        """Send email using SMTP.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text email content

        Returns:
            True if email was sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.config.sender_name} <{self.config.sender_email}>"
            msg["To"] = str(to_email)

            # Attach text and HTML parts
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")

            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()

                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def _get_password_reset_template(
        self, user_name: str, reset_link: str
    ) -> EmailTemplate:
        """Get password reset email template."""
        subject = "Password Reset Request - Pynomaly"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password Reset</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #007bff; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .button {{
                    display: inline-block;
                    padding: 12px 24px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    margin: 20px 0;
                }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Pynomaly</h1>
                    <p>Password Reset Request</p>
                </div>

                <div class="content">
                    <p>Hello {user_name},</p>

                    <p>We received a request to reset your password for your Pynomaly account. If you made this request, click the button below to reset your password:</p>

                    <p style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset Password</a>
                    </p>

                    <p>If the button doesn't work, you can also copy and paste the following link into your browser:</p>
                    <p style="word-break: break-all; background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
                        {reset_link}
                    </p>

                    <div class="warning">
                        <strong>Important:</strong> This link will expire in 1 hour for security reasons. If you didn't request this password reset, please ignore this email or contact support if you have concerns.
                    </div>

                    <p>Best regards,<br>The Pynomaly Team</p>
                </div>

                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                    <p>&copy; 2024 Pynomaly. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        Hello {user_name},

        We received a request to reset your password for your Pynomaly account.

        To reset your password, please visit the following link:
        {reset_link}

        IMPORTANT: This link will expire in 1 hour for security reasons.

        If you didn't request this password reset, please ignore this email or contact support if you have concerns.

        Best regards,
        The Pynomaly Team

        ---
        This is an automated message. Please do not reply to this email.
        """

        return EmailTemplate(
            subject=subject, html_content=html_content, text_content=text_content
        )

    def _get_user_invitation_template(
        self, inviter_name: str, organization_name: str, invitation_link: str
    ) -> EmailTemplate:
        """Get user invitation email template."""
        subject = f"You're invited to join {organization_name} on Pynomaly"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Invitation to Pynomaly</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #28a745; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .button {{
                    display: inline-block;
                    padding: 12px 24px;
                    background-color: #28a745;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    margin: 20px 0;
                }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                .features {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 20px 0; }}
                .features ul {{ margin: 0; padding-left: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Pynomaly</h1>
                    <p>You're Invited!</p>
                </div>

                <div class="content">
                    <p>Hello!</p>

                    <p>{inviter_name} has invited you to join <strong>{organization_name}</strong> on Pynomaly - the comprehensive anomaly detection platform.</p>

                    <div class="features">
                        <h3>With Pynomaly, you can:</h3>
                        <ul>
                            <li>Detect anomalies in your data using advanced ML algorithms</li>
                            <li>Monitor data quality and system health in real-time</li>
                            <li>Collaborate with your team on anomaly detection projects</li>
                            <li>Access detailed analytics and reporting</li>
                        </ul>
                    </div>

                    <p>Click the button below to accept the invitation and create your account:</p>

                    <p style="text-align: center;">
                        <a href="{invitation_link}" class="button">Accept Invitation</a>
                    </p>

                    <p>If the button doesn't work, you can also copy and paste the following link into your browser:</p>
                    <p style="word-break: break-all; background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
                        {invitation_link}
                    </p>

                    <p>This invitation will expire in 7 days.</p>

                    <p>Welcome to Pynomaly!</p>
                    <p>The Pynomaly Team</p>
                </div>

                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                    <p>&copy; 2024 Pynomaly. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        Hello!

        {inviter_name} has invited you to join {organization_name} on Pynomaly - the comprehensive anomaly detection platform.

        With Pynomaly, you can:
        - Detect anomalies in your data using advanced ML algorithms
        - Monitor data quality and system health in real-time
        - Collaborate with your team on anomaly detection projects
        - Access detailed analytics and reporting

        To accept this invitation and create your account, please visit:
        {invitation_link}

        This invitation will expire in 7 days.

        Welcome to Pynomaly!
        The Pynomaly Team

        ---
        This is an automated message. Please do not reply to this email.
        """

        return EmailTemplate(
            subject=subject, html_content=html_content, text_content=text_content
        )

    def _get_system_notification_template(
        self, subject: str, message: str, priority: str
    ) -> EmailTemplate:
        """Get system notification email template."""
        priority_colors = {
            "low": "#6c757d",
            "normal": "#007bff",
            "high": "#fd7e14",
            "urgent": "#dc3545",
        }

        priority_color = priority_colors.get(priority, "#007bff")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>System Notification</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {priority_color}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                .priority {{
                    background-color: {priority_color};
                    color: white;
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-size: 12px;
                    text-transform: uppercase;
                }}
                .message {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 4px;
                    margin: 20px 0;
                    border-left: 4px solid {priority_color};
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Pynomaly</h1>
                    <p>System Notification</p>
                </div>

                <div class="content">
                    <p><span class="priority">{priority} Priority</span></p>

                    <div class="message">
                        <p>{message}</p>
                    </div>

                    <p>This is an automated system notification from Pynomaly.</p>

                    <p>Best regards,<br>The Pynomaly System</p>
                </div>

                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                    <p>&copy; 2024 Pynomaly. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_content = f"""
        PYNOMALY SYSTEM NOTIFICATION

        Priority: {priority.upper()}

        {message}

        This is an automated system notification from Pynomaly.

        Best regards,
        The Pynomaly System

        ---
        This is an automated message. Please do not reply to this email.
        """

        return EmailTemplate(
            subject=subject, html_content=html_content, text_content=text_content
        )


# Global email service instance
_email_service: EmailService | None = None


def get_email_service() -> EmailService | None:
    """Get the global email service instance."""
    return _email_service


def init_email_service(settings: Settings) -> EmailService | None:
    """Initialize the global email service."""
    global _email_service

    try:
        # Check if email is configured
        if not all(
            [
                settings.security.smtp_server,
                settings.security.smtp_username,
                settings.security.smtp_password,
                settings.security.sender_email,
            ]
        ):
            logger.warning(
                "Email service not configured - email features will be disabled"
            )
            return None

        config = EmailConfig(
            smtp_server=settings.security.smtp_server,
            smtp_port=settings.security.smtp_port,
            smtp_username=settings.security.smtp_username,
            smtp_password=settings.security.smtp_password,
            use_tls=settings.security.smtp_use_tls,
            sender_email=settings.security.sender_email,
            sender_name=settings.security.sender_name,
            base_url=settings.security.base_url,
        )

        _email_service = EmailService(config)
        logger.info("Email service initialized successfully")
        return _email_service

    except Exception as e:
        logger.error(f"Failed to initialize email service: {e}")
        return None
