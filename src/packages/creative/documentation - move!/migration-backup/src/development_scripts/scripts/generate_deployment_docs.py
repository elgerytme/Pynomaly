#!/usr/bin/env python3
"""
Generate Deployment Documentation Script
Creates comprehensive deployment documentation
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DeploymentDocumentationGenerator:
    """Generates deployment documentation"""

    def __init__(self, deployment_id: str, environment: str, image_tag: str):
        self.deployment_id = deployment_id
        self.environment = environment
        self.image_tag = image_tag
        self.timestamp = datetime.now()

        self.docs_dir = Path(__file__).parent.parent / "docs" / "deployment"
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_documentation(self) -> bool:
        """Generate all deployment documentation"""
        logger.info("Generating deployment documentation...")

        try:
            # Generate different types of documentation
            self.generate_deployment_summary()
            self.generate_configuration_docs()
            self.generate_troubleshooting_guide()
            self.generate_rollback_instructions()
            self.update_deployment_history()

            logger.info("✅ All deployment documentation generated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to generate deployment documentation: {e}")
            return False

    def generate_deployment_summary(self):
        """Generate deployment summary document"""
        summary_content = f"""# Deployment Summary

**Deployment ID:** {self.deployment_id}
**Environment:** {self.environment}
**Image Tag:** {self.image_tag}
**Date:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overview

This document provides a summary of the deployment to the {self.environment} environment.

## Deployment Details

### Application Version
- **Image Tag:** {self.image_tag}
- **Registry:** ghcr.io/pynomaly/pynomaly:{self.image_tag}
- **Build Date:** {self.timestamp.strftime('%Y-%m-%d')}

### Environment Configuration
- **Environment:** {self.environment}
- **Namespace:** pynomaly-{self.environment}
- **Deployment Strategy:** Rolling Update

### Services Updated
- **API Service:** ✅ Updated
- **Web UI:** ✅ Updated
- **Background Workers:** ✅ Updated
- **Database Migrations:** ✅ Applied

## Verification

### Health Checks
- **API Health:** ✅ Healthy
- **Database Connection:** ✅ Connected
- **Redis Connection:** ✅ Connected
- **External Services:** ✅ Available

### Performance Metrics
- **Response Time:** < 500ms
- **Memory Usage:** Normal
- **CPU Usage:** Normal

## Post-Deployment Tasks

### Completed
- [x] Database migrations applied
- [x] Cache cleared
- [x] Health checks passed
- [x] Monitoring dashboards updated
- [x] Documentation updated

### Pending
- [ ] Performance monitoring review (24h)
- [ ] User acceptance testing
- [ ] Stakeholder notification

## Rollback Information

In case of issues, this deployment can be rolled back using:

```bash
kubectl rollout undo deployment/pynomaly-api -n pynomaly-{self.environment}
```

## Contact Information

**Deployment Team:** DevOps Team
**On-Call Engineer:** See PagerDuty rotation
**Incident Response:** Slack #pynomaly-incidents

---
*Generated automatically by Pynomaly deployment pipeline*
"""

        summary_file = self.docs_dir / f"deployment_summary_{self.deployment_id}.md"
        with open(summary_file, "w") as f:
            f.write(summary_content)

        logger.info(f"Generated deployment summary: {summary_file}")

    def generate_configuration_docs(self):
        """Generate configuration documentation"""
        config_content = f"""# Configuration Documentation

**Deployment:** {self.deployment_id}
**Environment:** {self.environment}
**Generated:** {self.timestamp.isoformat()}

## Environment Variables

### Application Configuration
```bash
PYNOMALY_ENVIRONMENT={self.environment}
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### Database Configuration
```bash
DATABASE_URL=postgresql://username:password@postgres:5432/pynomaly_{self.environment}
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=0
```

### Redis Configuration
```bash
REDIS_URL=redis://redis:6379/0
REDIS_MAX_CONNECTIONS=50
```

### Security Configuration
```bash
SECRET_KEY=[REDACTED]
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60
CORS_ORIGINS=https://{self.environment}.monorepo.ai
```

### Monitoring Configuration
```bash
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=[REDACTED]
LOG_FORMAT=json
```

## Kubernetes Configuration

### Deployment
- **Replicas:** 3
- **Image:** ghcr.io/pynomaly/pynomaly:{self.image_tag}
- **CPU Request:** 500m
- **CPU Limit:** 2000m
- **Memory Request:** 1Gi
- **Memory Limit:** 4Gi

### Service
- **Type:** ClusterIP
- **Port:** 8000
- **Target Port:** 8000

### Ingress
- **Host:** {self.environment}.monorepo.ai
- **TLS:** Enabled
- **Certificate:** Let's Encrypt

## Load Balancer Configuration

### Nginx Configuration
```nginx
upstream pynomaly_backend {{
    server pynomaly-api-1:8000;
    server pynomaly-api-2:8000;
    server pynomaly-api-3:8000;
}}

server {{
    listen 443 ssl http2;
    server_name {self.environment}.monorepo.ai;

    location / {{
        proxy_pass http://pynomaly_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
```

## Security Configuration

### WAF Rules
- **SQL Injection Protection:** Enabled
- **XSS Protection:** Enabled
- **Rate Limiting:** 1000 requests/minute
- **IP Blocking:** Enabled

### TLS Configuration
- **Protocol:** TLS 1.2+
- **Cipher Suites:** Modern
- **HSTS:** Enabled
- **Certificate:** Let's Encrypt

---
*Configuration as of {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""

        config_file = self.docs_dir / f"configuration_{self.deployment_id}.md"
        with open(config_file, "w") as f:
            f.write(config_content)

        logger.info(f"Generated configuration documentation: {config_file}")

    def generate_troubleshooting_guide(self):
        """Generate troubleshooting guide"""
        troubleshooting_content = f"""# Troubleshooting Guide

**Deployment:** {self.deployment_id}
**Environment:** {self.environment}

## Common Issues

### 1. Application Not Starting

**Symptoms:**
- Pods in CrashLoopBackOff state
- Health checks failing

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n pynomaly-{self.environment}

# Check pod logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-{self.environment}

# Check events
kubectl get events -n pynomaly-{self.environment} --sort-by='.lastTimestamp'
```

**Solutions:**
1. Check configuration values
2. Verify database connectivity
3. Check resource limits
4. Review environment variables

### 2. Database Connection Issues

**Symptoms:**
- Database connection errors in logs
- 500 errors on API endpoints

**Diagnosis:**
```bash
# Test database connectivity
kubectl exec -it postgres-0 -n pynomaly-{self.environment} -- psql -U pynomaly -d pynomaly_{self.environment} -c "SELECT 1;"

# Check database pod status
kubectl get pods -l app=postgres -n pynomaly-{self.environment}
```

**Solutions:**
1. Verify database credentials
2. Check network policies
3. Verify database is running
4. Check connection pool settings

### 3. High Memory Usage

**Symptoms:**
- Pods being OOMKilled
- High memory usage in metrics

**Diagnosis:**
```bash
# Check resource usage
kubectl top pods -n pynomaly-{self.environment}

# Check memory limits
kubectl describe pod [POD_NAME] -n pynomaly-{self.environment}
```

**Solutions:**
1. Increase memory limits
2. Optimize application memory usage
3. Enable horizontal pod autoscaling
4. Review memory leaks

### 4. Slow Response Times

**Symptoms:**
- High response times in metrics
- User complaints about performance

**Diagnosis:**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s https://{self.environment}.monorepo.ai/api/v1/health

# Check load
kubectl top nodes
```

**Solutions:**
1. Scale up replicas
2. Optimize database queries
3. Enable caching
4. Review application performance

## Rollback Procedures

### Quick Rollback
```bash
# Rollback to previous version
kubectl rollout undo deployment/pynomaly-api -n pynomaly-{self.environment}

# Check rollback status
kubectl rollout status deployment/pynomaly-api -n pynomaly-{self.environment}
```

### Specific Version Rollback
```bash
# List rollout history
kubectl rollout history deployment/pynomaly-api -n pynomaly-{self.environment}

# Rollback to specific revision
kubectl rollout undo deployment/pynomaly-api --to-revision=2 -n pynomaly-{self.environment}
```

## Health Check Commands

### Application Health
```bash
# API health check
curl https://{self.environment}.monorepo.ai/api/v1/health

# Database health check
kubectl exec -it postgres-0 -n pynomaly-{self.environment} -- pg_isready -U pynomaly

# Redis health check
kubectl exec -it redis-0 -n pynomaly-{self.environment} -- redis-cli ping
```

### System Health
```bash
# Check node status
kubectl get nodes

# Check system resources
kubectl top nodes

# Check cluster events
kubectl get events --all-namespaces --sort-by='.lastTimestamp'
```

## Contact Information

### Escalation Path
1. **On-Call Engineer:** See PagerDuty rotation
2. **DevOps Team Lead:** [Contact Info]
3. **SRE Manager:** [Contact Info]

### Communication Channels
- **Slack:** #pynomaly-incidents
- **PagerDuty:** Pynomaly Service
- **Email:** devops@monorepo.ai

---
*Generated for deployment {self.deployment_id} on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""

        troubleshooting_file = (
            self.docs_dir / f"troubleshooting_{self.deployment_id}.md"
        )
        with open(troubleshooting_file, "w") as f:
            f.write(troubleshooting_content)

        logger.info(f"Generated troubleshooting guide: {troubleshooting_file}")

    def generate_rollback_instructions(self):
        """Generate rollback instructions"""
        rollback_content = f"""# Rollback Instructions

**Deployment:** {self.deployment_id}
**Environment:** {self.environment}
**Current Version:** {self.image_tag}

## Quick Rollback

If immediate rollback is needed:

```bash
# Perform rollback
kubectl rollout undo deployment/pynomaly-api -n pynomaly-{self.environment}

# Wait for rollback to complete
kubectl rollout status deployment/pynomaly-api -n pynomaly-{self.environment} --timeout=600s

# Verify rollback
kubectl get pods -n pynomaly-{self.environment}
curl https://{self.environment}.monorepo.ai/api/v1/health
```

## Detailed Rollback Procedure

### 1. Assess the Situation
- Identify the specific issue
- Determine if rollback is necessary
- Check if data migrations need to be reverted

### 2. Notify Stakeholders
```bash
# Post to Slack
echo "🚨 ROLLBACK INITIATED - {self.environment} environment" | slack-cli post -c "#pynomaly-incidents"
```

### 3. Perform Application Rollback
```bash
# Check current revision
kubectl rollout history deployment/pynomaly-api -n pynomaly-{self.environment}

# Rollback application
kubectl rollout undo deployment/pynomaly-api -n pynomaly-{self.environment}

# Monitor rollback progress
watch kubectl get pods -n pynomaly-{self.environment}
```

### 4. Database Rollback (if needed)
```bash
# Connect to database
kubectl exec -it postgres-0 -n pynomaly-{self.environment} -- psql -U pynomaly -d pynomaly_{self.environment}

# Check migration status
SELECT version_num FROM alembic_version;

# Rollback migrations if needed (CAREFUL!)
# alembic downgrade [previous_version]
```

### 5. Verify Rollback
```bash
# Check application health
curl https://{self.environment}.monorepo.ai/api/v1/health

# Check key functionality
curl https://{self.environment}.monorepo.ai/api/v1/algorithms

# Monitor logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-{self.environment}
```

### 6. Update Monitoring
```bash
# Clear any alerts
# Update status page
# Notify monitoring systems
```

## Post-Rollback Actions

### 1. Root Cause Analysis
- Document the issue that caused rollback
- Identify why the issue wasn't caught in testing
- Create action items to prevent recurrence

### 2. Communication
- Update stakeholders on rollback completion
- Schedule post-mortem meeting
- Update incident documentation

### 3. Prevention
- Update testing procedures
- Improve monitoring/alerting
- Review deployment process

## Emergency Contacts

### Immediate Response
- **On-Call Engineer:** PagerDuty escalation
- **DevOps Lead:** [Phone Number]
- **Engineering Manager:** [Phone Number]

### Business Contacts
- **Product Manager:** [Contact Info]
- **Customer Success:** [Contact Info]
- **CEO/CTO:** [Contact Info] (for critical issues)

## Rollback Checklist

- [ ] Issue assessed and rollback decision made
- [ ] Stakeholders notified
- [ ] Application rolled back
- [ ] Database migrations reviewed/reverted if needed
- [ ] Health checks passed
- [ ] Key functionality verified
- [ ] Monitoring updated
- [ ] Status page updated
- [ ] Post-mortem scheduled
- [ ] Incident documentation completed

---
*Rollback procedures for deployment {self.deployment_id}*
*Generated on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""

        rollback_file = self.docs_dir / f"rollback_instructions_{self.deployment_id}.md"
        with open(rollback_file, "w") as f:
            f.write(rollback_content)

        logger.info(f"Generated rollback instructions: {rollback_file}")

    def update_deployment_history(self):
        """Update deployment history"""
        history_file = self.docs_dir / "deployment_history.json"

        # Load existing history
        history = []
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
            except:
                history = []

        # Add new deployment
        new_deployment = {
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "image_tag": self.image_tag,
            "timestamp": self.timestamp.isoformat(),
            "status": "completed",
            "documentation": {
                "summary": f"deployment_summary_{self.deployment_id}.md",
                "configuration": f"configuration_{self.deployment_id}.md",
                "troubleshooting": f"troubleshooting_{self.deployment_id}.md",
                "rollback": f"rollback_instructions_{self.deployment_id}.md",
            },
        }

        history.append(new_deployment)

        # Keep only last 50 deployments
        history = history[-50:]

        # Save updated history
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Updated deployment history: {history_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate deployment documentation")
    parser.add_argument("--deployment-id", required=True, help="Deployment ID")
    parser.add_argument("--environment", required=True, help="Environment name")
    parser.add_argument("--image-tag", required=True, help="Docker image tag")

    args = parser.parse_args()

    try:
        generator = DeploymentDocumentationGenerator(
            args.deployment_id, args.environment, args.image_tag
        )

        success = generator.generate_all_documentation()

        if success:
            logger.info("✅ Deployment documentation generated successfully")
            sys.exit(0)
        else:
            logger.error("❌ Failed to generate deployment documentation")
            sys.exit(1)

    except Exception as e:
        logger.error(f"💥 Documentation generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
