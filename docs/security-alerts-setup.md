# Security Alerts and Notifications Setup

This document explains how to set up the security alerts and notifications system for the Pynomaly project.

## Overview

The security alerts system monitors GitHub Code Scanning alerts and sends notifications when new HIGH or CRITICAL severity issues are detected. It supports both Slack notifications and email fallback.

## Features

- 🔍 **Automated Monitoring**: Runs daily to check for new HIGH/CRITICAL security alerts
- 📱 **Slack Integration**: Primary notification method via Slack webhook
- 📧 **Email Fallback**: Automatic email notifications when Slack is unavailable
- 🎯 **Smart Filtering**: Only notifies about new alerts (created in last 24 hours)
- 📊 **Rich Notifications**: Includes alert details, links, and statistics
- 🔧 **Manual Triggers**: Can be triggered manually with force notification option

## Required Secrets

### Slack Webhook (Primary)
Add the following secret to your repository:

```
SLACK_WEBHOOK_URL
```

**How to get a Slack webhook URL:**
1. Go to https://api.slack.com/apps
2. Create a new app or use an existing one
3. Navigate to "Incoming Webhooks"
4. Activate incoming webhooks
5. Create a new webhook for your desired channel
6. Copy the webhook URL

### Email Configuration (Fallback)
Add these secrets for email notifications:

```
NOTIFICATION_EMAIL    # Email address to send notifications to
FROM_EMAIL           # Email address to send from (optional)
SMTP_SERVER          # SMTP server hostname (optional)
SMTP_PORT            # SMTP server port (optional, default: 587)
SMTP_USERNAME        # SMTP authentication username (optional)
SMTP_PASSWORD        # SMTP authentication password (optional)
```

## Setting Up Secrets

1. Go to your repository on GitHub
2. Navigate to **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Add each secret with its corresponding value

### Example Secret Configuration

```
SLACK_WEBHOOK_URL: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
NOTIFICATION_EMAIL: security-team@yourcompany.com
FROM_EMAIL: github-actions@yourcompany.com
SMTP_SERVER: smtp.gmail.com
SMTP_PORT: 587
SMTP_USERNAME: your-smtp-username
SMTP_PASSWORD: your-smtp-password
```

## How It Works

### Workflow Trigger
- **Scheduled**: Runs daily at 2 AM UTC
- **Manual**: Can be triggered manually with optional force notification
- **Event-based**: Monitors for new alerts created in the last 24 hours

### Notification Logic
1. **Check for Slack webhook**: Verifies if `SLACK_WEBHOOK_URL` secret exists
2. **Fetch security alerts**: Gets all open HIGH/CRITICAL alerts from GitHub Code Scanning
3. **Filter new alerts**: Only considers alerts created in the last 24 hours
4. **Send notifications**:
   - If Slack webhook is available: Send rich Slack notification
   - If Slack webhook is missing: Send email as fallback
   - If no new alerts: No notification sent (unless forced)

### Email Delivery Methods
The system tries multiple email delivery methods:
1. **sendmail**: Standard Unix mail transfer agent
2. **mail command**: Alternative mail utility
3. **curl SMTP**: Direct SMTP connection using curl

## Notification Content

### Slack Notification
- Rich formatted message with buttons
- Repository information
- Alert statistics (new vs total)
- Direct link to view alerts

### Email Notification
- Plain text format
- Detailed alert information
- File locations and line numbers
- Direct links to alerts

## Troubleshooting

### Common Issues

1. **No notifications received**
   - Check if secrets are properly configured
   - Verify webhook URL is correct
   - Check workflow logs for errors

2. **Slack notifications not working**
   - Verify `SLACK_WEBHOOK_URL` secret is set
   - Test webhook URL manually
   - Check Slack app permissions

3. **Email notifications failing**
   - Verify email secrets are configured
   - Check SMTP server settings
   - Review workflow logs for delivery errors

### Testing

To test the notification system:
1. Go to **Actions** tab in your repository
2. Find "Security Alerts and Notifications" workflow
3. Click "Run workflow"
4. Enable "Force send notification"
5. Click "Run workflow" button

## Monitoring

The workflow provides detailed logs including:
- Alert discovery and filtering
- Notification delivery status
- Error details if notifications fail
- Summary of actions taken

## Customization

### Modifying Alert Thresholds
Edit the workflow file to change severity levels:
```yaml
# Current: HIGH and CRITICAL
alert.rule.severity === 'high' || alert.rule.severity === 'critical'

# Example: Include MEDIUM
alert.rule.severity === 'medium' || alert.rule.severity === 'high' || alert.rule.severity === 'critical'
```

### Changing Schedule
Modify the cron expression in the workflow:
```yaml
schedule:
  # Current: Daily at 2 AM UTC
  - cron: '0 2 * * *'
  # Example: Every 6 hours
  - cron: '0 */6 * * *'
```

### Custom Email Templates
Modify the email body in the workflow script to customize format and content.

## Security Considerations

- All secrets are stored securely in GitHub Secrets
- Webhook URLs and credentials are never exposed in logs
- Email content is logged only in case of delivery failures
- SMTP passwords are masked in all outputs

## Support

For issues with the alerting system:
1. Check workflow logs in the Actions tab
2. Verify secret configuration
3. Review this documentation
4. Contact the security team if issues persist
