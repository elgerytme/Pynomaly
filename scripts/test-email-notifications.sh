#!/bin/bash

# Test script for email notifications
# This script helps test the email notification functionality locally

set -e

echo "🧪 Testing Email Notification System"
echo "===================================="

# Check if required tools are installed
echo "Checking system requirements..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for sendmail
if command_exists sendmail; then
    echo "✅ sendmail is available"
    SENDMAIL_AVAILABLE=true
else
    echo "❌ sendmail is not available"
    SENDMAIL_AVAILABLE=false
fi

# Check for mail command
if command_exists mail; then
    echo "✅ mail command is available"
    MAIL_AVAILABLE=true
else
    echo "❌ mail command is not available"
    MAIL_AVAILABLE=false
fi

# Check for curl
if command_exists curl; then
    echo "✅ curl is available"
    CURL_AVAILABLE=true
else
    echo "❌ curl is not available"
    CURL_AVAILABLE=false
fi

echo ""

# Email configuration
echo "📧 Email Configuration"
echo "---------------------"
TO_EMAIL=${NOTIFICATION_EMAIL:-"admin@example.com"}
FROM_EMAIL=${FROM_EMAIL:-"github-actions@example.com"}
SMTP_SERVER=${SMTP_SERVER:-"localhost"}
SMTP_PORT=${SMTP_PORT:-"587"}
SMTP_USERNAME=${SMTP_USERNAME:-""}
SMTP_PASSWORD=${SMTP_PASSWORD:-""}

echo "To: $TO_EMAIL"
echo "From: $FROM_EMAIL"
echo "SMTP Server: $SMTP_SERVER:$SMTP_PORT"
if [ -n "$SMTP_USERNAME" ]; then
    echo "SMTP Username: $SMTP_USERNAME"
    echo "SMTP Password: [SET]"
else
    echo "SMTP Auth: Not configured"
fi

echo ""

# Create test email content
SUBJECT="🧪 Test Security Alert - $(date)"
EMAIL_BODY="Subject: $SUBJECT
From: $FROM_EMAIL
To: $TO_EMAIL
Content-Type: text/plain; charset=utf-8

🚨 Security Alert Summary

This is a test email from the security notification system.

Repository: test/repository
Scan Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')

📊 Alert Statistics:
• Total HIGH/CRITICAL alerts: 5
• New alerts (last 24h): 2

🔍 New Alert Details:
1. [HIGH] Potential SQL Injection
   File: src/database.py:42
   URL: https://github.com/test/repository/security/code-scanning/1
   Created: $(date)

2. [CRITICAL] Hardcoded Secret
   File: config/settings.py:15
   URL: https://github.com/test/repository/security/code-scanning/2
   Created: $(date)

🔗 View all alerts: https://github.com/test/repository/security/code-scanning

---
This is an automated test notification from GitHub Actions.
Repository: https://github.com/test/repository
"

# Create temporary file
TEMP_FILE=$(mktemp)
echo "$EMAIL_BODY" > "$TEMP_FILE"

echo "📝 Test Email Content"
echo "--------------------"
echo "Subject: $SUBJECT"
echo "Content preview:"
echo "$EMAIL_BODY" | head -10
echo "..."
echo ""

# Test email delivery methods
echo "🚀 Testing Email Delivery Methods"
echo "--------------------------------"

SUCCESS=false

# Method 1: sendmail
if [ "$SENDMAIL_AVAILABLE" = true ]; then
    echo "Testing sendmail..."
    if sendmail -t < "$TEMP_FILE" 2>/dev/null; then
        echo "✅ sendmail test successful"
        SUCCESS=true
    else
        echo "❌ sendmail test failed"
    fi
else
    echo "⏭️  Skipping sendmail (not available)"
fi

# Method 2: mail command
if [ "$MAIL_AVAILABLE" = true ] && [ "$SUCCESS" = false ]; then
    echo "Testing mail command..."
    if echo "$EMAIL_BODY" | mail -s "$SUBJECT" "$TO_EMAIL" 2>/dev/null; then
        echo "✅ mail command test successful"
        SUCCESS=true
    else
        echo "❌ mail command test failed"
    fi
else
    echo "⏭️  Skipping mail command (not available or previous method succeeded)"
fi

# Method 3: curl SMTP
if [ "$CURL_AVAILABLE" = true ] && [ "$SUCCESS" = false ] && [ "$SMTP_SERVER" != "localhost" ]; then
    echo "Testing curl SMTP..."
    
    CURL_CMD="curl -s --mail-from \"$FROM_EMAIL\" --mail-rcpt \"$TO_EMAIL\" --upload-file \"$TEMP_FILE\" \"smtp://$SMTP_SERVER:$SMTP_PORT\""
    
    if [ -n "$SMTP_USERNAME" ] && [ -n "$SMTP_PASSWORD" ]; then
        CURL_CMD="$CURL_CMD --user \"$SMTP_USERNAME:$SMTP_PASSWORD\""
    fi
    
    if eval "$CURL_CMD" 2>/dev/null; then
        echo "✅ curl SMTP test successful"
        SUCCESS=true
    else
        echo "❌ curl SMTP test failed"
    fi
else
    echo "⏭️  Skipping curl SMTP (not available, previous method succeeded, or localhost SMTP)"
fi

# Cleanup
rm -f "$TEMP_FILE"

echo ""
echo "📊 Test Summary"
echo "==============")
if [ "$SUCCESS" = true ]; then
    echo "✅ At least one email delivery method works"
    echo "The notification system should work properly"
else
    echo "❌ No email delivery methods are working"
    echo "Please check your email configuration"
fi

echo ""
echo "💡 Next Steps"
echo "============"
echo "1. Add required secrets to your GitHub repository:"
echo "   - SLACK_WEBHOOK_URL (for Slack notifications)"
echo "   - NOTIFICATION_EMAIL (for email notifications)"
echo "   - SMTP configuration secrets (if using external SMTP)"
echo ""
echo "2. Test the workflow manually:"
echo "   - Go to Actions tab in your repository"
echo "   - Run 'Security Alerts and Notifications' workflow"
echo "   - Enable 'Force send notification' option"
echo ""
echo "3. Monitor the workflow logs for any issues"
echo ""
echo "For more information, see docs/security-alerts-setup.md"
