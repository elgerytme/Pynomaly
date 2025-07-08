# Test script for email notifications (PowerShell version)
# This script helps test the email notification functionality on Windows

param(
    [string]$ToEmail = $env:NOTIFICATION_EMAIL,
    [string]$FromEmail = $env:FROM_EMAIL,
    [string]$SmtpServer = $env:SMTP_SERVER,
    [int]$SmtpPort = [int]$env:SMTP_PORT,
    [string]$SmtpUsername = $env:SMTP_USERNAME,
    [string]$SmtpPassword = $env:SMTP_PASSWORD
)

Write-Host "🧪 Testing Email Notification System" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Set default values if not provided
if (-not $ToEmail) { $ToEmail = "admin@example.com" }
if (-not $FromEmail) { $FromEmail = "github-actions@example.com" }
if (-not $SmtpServer) { $SmtpServer = "localhost" }
if (-not $SmtpPort) { $SmtpPort = 587 }

Write-Host ""
Write-Host "📧 Email Configuration" -ForegroundColor Yellow
Write-Host "---------------------" -ForegroundColor Yellow
Write-Host "To: $ToEmail"
Write-Host "From: $FromEmail"
Write-Host "SMTP Server: $SmtpServer`:$SmtpPort"
if ($SmtpUsername) {
    Write-Host "SMTP Username: $SmtpUsername"
    Write-Host "SMTP Password: [SET]"
} else {
    Write-Host "SMTP Auth: Not configured"
}

Write-Host ""

# Create test email content
$Subject = "🧪 Test Security Alert - $(Get-Date)"
$Body = @"
🚨 Security Alert Summary

This is a test email from the security notification system.

Repository: test/repository
Scan Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')

📊 Alert Statistics:
• Total HIGH/CRITICAL alerts: 5
• New alerts (last 24h): 2

🔍 New Alert Details:
1. [HIGH] Potential SQL Injection
   File: src/database.py:42
   URL: https://github.com/test/repository/security/code-scanning/1
   Created: $(Get-Date)

2. [CRITICAL] Hardcoded Secret
   File: config/settings.py:15
   URL: https://github.com/test/repository/security/code-scanning/2
   Created: $(Get-Date)

🔗 View all alerts: https://github.com/test/repository/security/code-scanning

---
This is an automated test notification from GitHub Actions.
Repository: https://github.com/test/repository
"@

Write-Host "📝 Test Email Content" -ForegroundColor Yellow
Write-Host "--------------------" -ForegroundColor Yellow
Write-Host "Subject: $Subject"
Write-Host "Content preview:"
Write-Host ($Body -split "`n" | Select-Object -First 10 | Out-String)
Write-Host "..."
Write-Host ""

# Test email delivery
Write-Host "🚀 Testing Email Delivery" -ForegroundColor Yellow
Write-Host "------------------------" -ForegroundColor Yellow

$Success = $false

try {
    if ($SmtpServer -ne "localhost" -and $SmtpUsername -and $SmtpPassword) {
        # Use authenticated SMTP
        Write-Host "Testing authenticated SMTP..."
        $SecurePassword = ConvertTo-SecureString $SmtpPassword -AsPlainText -Force
        $Credential = New-Object System.Management.Automation.PSCredential($SmtpUsername, $SecurePassword)
        
        Send-MailMessage -To $ToEmail -From $FromEmail -Subject $Subject -Body $Body -SmtpServer $SmtpServer -Port $SmtpPort -Credential $Credential -UseSsl
        Write-Host "✅ Authenticated SMTP test successful" -ForegroundColor Green
        $Success = $true
    }
    elseif ($SmtpServer -ne "localhost") {
        # Use non-authenticated SMTP
        Write-Host "Testing non-authenticated SMTP..."
        Send-MailMessage -To $ToEmail -From $FromEmail -Subject $Subject -Body $Body -SmtpServer $SmtpServer -Port $SmtpPort
        Write-Host "✅ Non-authenticated SMTP test successful" -ForegroundColor Green
        $Success = $true
    }
    else {
        Write-Host "⏭️  Skipping SMTP test (localhost configuration)" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ SMTP test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "📊 Test Summary" -ForegroundColor Yellow
Write-Host "==============" -ForegroundColor Yellow
if ($Success) {
    Write-Host "✅ Email delivery test successful" -ForegroundColor Green
    Write-Host "The notification system should work properly" -ForegroundColor Green
} else {
    Write-Host "❌ Email delivery test failed" -ForegroundColor Red
    Write-Host "Please check your email configuration" -ForegroundColor Red
}

Write-Host ""
Write-Host "💡 Next Steps" -ForegroundColor Yellow
Write-Host "============" -ForegroundColor Yellow
Write-Host "1. Add required secrets to your GitHub repository:"
Write-Host "   - SLACK_WEBHOOK_URL (for Slack notifications)"
Write-Host "   - NOTIFICATION_EMAIL (for email notifications)"
Write-Host "   - SMTP configuration secrets (if using external SMTP)"
Write-Host ""
Write-Host "2. Test the workflow manually:"
Write-Host "   - Go to Actions tab in your repository"
Write-Host "   - Run 'Security Alerts and Notifications' workflow"
Write-Host "   - Enable 'Force send notification' option"
Write-Host ""
Write-Host "3. Monitor the workflow logs for any issues"
Write-Host ""
Write-Host "For more information, see docs/security-alerts-setup.md"
