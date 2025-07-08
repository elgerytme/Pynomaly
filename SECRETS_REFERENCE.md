# GitHub Secrets Reference for Security Alerts

## Required Secrets

### Primary Notification (Slack)
```
SLACK_WEBHOOK_URL
```
**Description**: Slack webhook URL for sending notifications  
**Required**: Yes (for Slack notifications)  
**Example**: `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX`

### Fallback Notification (Email)
```
NOTIFICATION_EMAIL
```
**Description**: Email address to receive security alerts  
**Required**: Yes (for email fallback)  
**Example**: `security-team@yourcompany.com`

### Optional Email Configuration
```
FROM_EMAIL
```
**Description**: Email address to send from  
**Required**: No  
**Default**: `github-actions@example.com`  
**Example**: `notifications@yourcompany.com`

```
SMTP_SERVER
```
**Description**: SMTP server hostname  
**Required**: No (uses sendmail/mail if not provided)  
**Example**: `smtp.gmail.com`

```
SMTP_PORT
```
**Description**: SMTP server port  
**Required**: No  
**Default**: `587`  
**Example**: `587` (for TLS) or `465` (for SSL)

```
SMTP_USERNAME
```
**Description**: SMTP authentication username  
**Required**: No (only if SMTP auth is needed)  
**Example**: `your-email@gmail.com`

```
SMTP_PASSWORD
```
**Description**: SMTP authentication password  
**Required**: No (only if SMTP auth is needed)  
**Example**: `your-app-password`

## How to Add Secrets

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the secret name and value
5. Click **Add secret**

## Minimal Setup

### Option 1: Slack Only
```
SLACK_WEBHOOK_URL = https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

### Option 2: Email Only
```
NOTIFICATION_EMAIL = admin@yourcompany.com
```

### Option 3: Both Slack and Email
```
SLACK_WEBHOOK_URL = https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
NOTIFICATION_EMAIL = admin@yourcompany.com
```

## Example Configurations

### Gmail SMTP
```
NOTIFICATION_EMAIL = recipient@gmail.com
FROM_EMAIL = sender@gmail.com
SMTP_SERVER = smtp.gmail.com
SMTP_PORT = 587
SMTP_USERNAME = sender@gmail.com
SMTP_PASSWORD = your-app-password
```

### Office 365 SMTP
```
NOTIFICATION_EMAIL = recipient@yourcompany.com
FROM_EMAIL = sender@yourcompany.com
SMTP_SERVER = smtp.office365.com
SMTP_PORT = 587
SMTP_USERNAME = sender@yourcompany.com
SMTP_PASSWORD = your-password
```

### SendGrid SMTP
```
NOTIFICATION_EMAIL = recipient@yourcompany.com
FROM_EMAIL = sender@yourcompany.com
SMTP_SERVER = smtp.sendgrid.net
SMTP_PORT = 587
SMTP_USERNAME = apikey
SMTP_PASSWORD = your-sendgrid-api-key
```

## Testing

### Test Email Configuration (PowerShell)
```powershell
.\scripts\test-email-notifications.ps1 -ToEmail "test@example.com" -SmtpServer "smtp.gmail.com" -SmtpUsername "your-email@gmail.com" -SmtpPassword "your-password"
```

### Test Email Configuration (Bash)
```bash
export NOTIFICATION_EMAIL="test@example.com"
export SMTP_SERVER="smtp.gmail.com"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-password"
./scripts/test-email-notifications.sh
```

### Test in GitHub Actions
1. Go to **Actions** tab
2. Select **Security Alerts and Notifications** workflow
3. Click **Run workflow**
4. Check **Force send notification**
5. Click **Run workflow**

## Troubleshooting

### Slack Issues
- Verify webhook URL is correct
- Check Slack app permissions
- Test webhook manually with curl

### Email Issues
- Check SMTP server settings
- Verify credentials are correct
- Test with local email client first
- Check firewall/network restrictions

### GitHub Actions Issues
- Verify secrets are added correctly
- Check workflow logs for errors
- Ensure proper permissions are set
- Test with force notification first

## Security Notes

- Never commit secrets to your repository
- Use GitHub Secrets for all sensitive data
- Rotate webhook URLs and passwords regularly
- Monitor access logs for suspicious activity
- Use app passwords instead of account passwords where possible
