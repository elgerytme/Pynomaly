# Pynomaly IAM Policy Guide & Least-Privilege Security

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🔒 [Security](README.md) > 📄 IAM Guide

---

## Overview

This guide provides comprehensive IAM policy examples and least-privilege recommendations specifically designed for the Pynomaly anomaly detection platform. These policies ensure secure access control while maintaining operational efficiency.

## Table of Contents

- [Core Principles](#core-principles)
- [Role-Based Policies](#role-based-policies)
- [Service-Specific Policies](#service-specific-policies)
- [API Access Policies](#api-access-policies)
- [Data Access Policies](#data-access-policies)
- [Deployment Policies](#deployment-policies)
- [Best Practices](#best-practices)
- [Implementation Examples](#implementation-examples)

## Core Principles

### 1. Least Privilege Access
Grant only the minimum permissions required for users to perform their job functions.

### 2. Role-Based Access Control (RBAC)
Organize permissions into logical roles that align with organizational responsibilities.

### 3. Defense in Depth
Implement multiple layers of security controls and validation.

### 4. Regular Auditing
Continuously monitor and review access patterns and permissions.

## Role-Based Policies

### Administrator Role
Full platform administration with all permissions.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:*"
      ],
      "Resource": "*",
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": ["203.0.113.0/24", "198.51.100.0/24"]
        },
        "Bool": {
          "aws:MultiFactorAuthPresent": "true"
        }
      }
    }
  ]
}
```

### Data Scientist Role
Focused on model development and experimentation.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:datasets:read",
        "pynomaly:datasets:create",
        "pynomaly:detectors:read",
        "pynomaly:detectors:create",
        "pynomaly:detectors:train",
        "pynomaly:detectors:predict",
        "pynomaly:experiments:*",
        "pynomaly:models:read",
        "pynomaly:models:create",
        "pynomaly:automl:*"
      ],
      "Resource": [
        "arn:aws:pynomaly:*:*:dataset/*",
        "arn:aws:pynomaly:*:*:detector/*",
        "arn:aws:pynomaly:*:*:experiment/*",
        "arn:aws:pynomaly:*:*:model/*"
      ],
      "Condition": {
        "StringEquals": {
          "pynomaly:department": "${aws:PrincipalTag/department}"
        }
      }
    },
    {
      "Effect": "Deny",
      "Action": [
        "pynomaly:datasets:delete",
        "pynomaly:detectors:delete",
        "pynomaly:users:*",
        "pynomaly:admin:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### Analyst Role
Read-only access for data analysis and reporting.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:datasets:read",
        "pynomaly:detectors:read",
        "pynomaly:detectors:predict",
        "pynomaly:results:read",
        "pynomaly:reports:read",
        "pynomaly:reports:create",
        "pynomaly:dashboards:read"
      ],
      "Resource": "*",
      "Condition": {
        "StringLike": {
          "pynomaly:dataset-classification": ["public", "internal"]
        }
      }
    },
    {
      "Effect": "Deny",
      "Action": [
        "pynomaly:*:create",
        "pynomaly:*:update",
        "pynomaly:*:delete"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "pynomaly:dataset-classification": "confidential"
        }
      }
    }
  ]
}
```

### API Service Role
For programmatic access and automation.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:detectors:predict",
        "pynomaly:datasets:read",
        "pynomaly:streaming:*"
      ],
      "Resource": [
        "arn:aws:pynomaly:*:*:detector/production-*",
        "arn:aws:pynomaly:*:*:dataset/production-*"
      ],
      "Condition": {
        "StringEquals": {
          "pynomaly:environment": "production"
        },
        "DateGreaterThan": {
          "aws:CurrentTime": "2024-01-01T00:00:00Z"
        },
        "DateLessThan": {
          "aws:CurrentTime": "2025-01-01T00:00:00Z"
        }
      }
    }
  ]
}
```

## Service-Specific Policies

### Model Training Service
Permissions for automated training pipelines.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:detectors:create",
        "pynomaly:detectors:train",
        "pynomaly:detectors:update",
        "pynomaly:models:create",
        "pynomaly:models:update",
        "pynomaly:experiments:create",
        "pynomaly:experiments:update"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "pynomaly:training-role": "automated"
        },
        "NumericLessThan": {
          "pynomaly:training-duration-hours": "24"
        }
      }
    }
  ]
}
```

### Data Ingestion Service
Permissions for data pipeline operations.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:datasets:create",
        "pynomaly:datasets:update",
        "pynomaly:datasets:validate",
        "pynomaly:preprocessing:*"
      ],
      "Resource": [
        "arn:aws:pynomaly:*:*:dataset/ingestion-*"
      ],
      "Condition": {
        "StringEquals": {
          "pynomaly:source-system": "${aws:PrincipalTag/source-system}"
        }
      }
    }
  ]
}
```

## API Access Policies

### REST API Access
Controlled access to REST endpoints.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:api:get",
        "pynomaly:api:post"
      ],
      "Resource": [
        "arn:aws:pynomaly:*:*:api/v1/detection/*",
        "arn:aws:pynomaly:*:*:api/v1/datasets/*"
      ],
      "Condition": {
        "StringEquals": {
          "pynomaly:api-version": "v1"
        },
        "NumericLessThan": {
          "pynomaly:request-rate": "100"
        }
      }
    }
  ]
}
```

### WebSocket Access
Real-time streaming permissions.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:websocket:connect",
        "pynomaly:streaming:subscribe",
        "pynomaly:streaming:publish"
      ],
      "Resource": [
        "arn:aws:pynomaly:*:*:stream/anomaly-alerts",
        "arn:aws:pynomaly:*:*:stream/training-status"
      ],
      "Condition": {
        "StringEquals": {
          "pynomaly:connection-type": "secure"
        }
      }
    }
  ]
}
```

## Data Access Policies

### Confidential Data Access
Strict controls for sensitive data.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:datasets:read",
        "pynomaly:datasets:decrypt"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "pynomaly:data-classification": "confidential",
          "pynomaly:encryption-key-owner": "${aws:userid}"
        },
        "Bool": {
          "aws:MultiFactorAuthPresent": "true"
        },
        "IpAddress": {
          "aws:SourceIp": ["203.0.113.0/24"]
        }
      }
    }
  ]
}
```

### Public Data Access
Broader access for non-sensitive data.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:datasets:read",
        "pynomaly:datasets:list"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "pynomaly:data-classification": "public"
        }
      }
    }
  ]
}
```

## Deployment Policies

### Development Environment
Flexible permissions for development work.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:*"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "pynomaly:environment": "development"
        },
        "StringLike": {
          "aws:userid": "*:${aws:username}"
        }
      }
    },
    {
      "Effect": "Deny",
      "Action": [
        "pynomaly:production:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### Production Environment
Restricted access for production systems.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "pynomaly:detectors:predict",
        "pynomaly:monitoring:read",
        "pynomaly:health:check"
      ],
      "Resource": [
        "arn:aws:pynomaly:*:*:detector/prod-*"
      ],
      "Condition": {
        "StringEquals": {
          "pynomaly:environment": "production"
        },
        "Bool": {
          "aws:SecureTransport": "true"
        }
      }
    }
  ]
}
```

## Best Practices

### 1. Use Policy Conditions
Always include appropriate conditions to limit scope:

```json
{
  "Condition": {
    "StringEquals": {
      "pynomaly:environment": "development"
    },
    "IpAddress": {
      "aws:SourceIp": ["203.0.113.0/24"]
    },
    "Bool": {
      "aws:MultiFactorAuthPresent": "true"
    },
    "DateGreaterThan": {
      "aws:CurrentTime": "2024-01-01T00:00:00Z"
    }
  }
}
```

### 2. Resource-Specific Permissions
Use specific resource ARNs instead of wildcards:

```json
{
  "Resource": [
    "arn:aws:pynomaly:us-west-2:123456789012:dataset/prod-*",
    "arn:aws:pynomaly:us-west-2:123456789012:detector/prod-*"
  ]
}
```

### 3. Time-Bound Access
Include time restrictions for temporary access:

```json
{
  "Condition": {
    "DateGreaterThan": {
      "aws:CurrentTime": "2024-01-01T00:00:00Z"
    },
    "DateLessThan": {
      "aws:CurrentTime": "2024-12-31T23:59:59Z"
    }
  }
}
```

### 4. Request Rate Limiting
Control API usage with rate limiting:

```json
{
  "Condition": {
    "NumericLessThan": {
      "pynomaly:requests-per-minute": "100"
    }
  }
}
```

## Implementation Examples

### Environment Setup
Configure different permission sets per environment:

```bash
# Development Environment
export PYNOMALY_ENVIRONMENT=development
export PYNOMALY_IAM_ROLE=arn:aws:iam::123456789012:role/PynomagyDeveloperRole

# Production Environment  
export PYNOMALY_ENVIRONMENT=production
export PYNOMALY_IAM_ROLE=arn:aws:iam::123456789012:role/PynomagyProductionRole
```

### Policy Validation
Test policies before deployment:

```python
import boto3
from pynomaly.security import validate_iam_policy

# Validate policy syntax and permissions
policy = {
    "Version": "2012-10-17",
    "Statement": [...]
}

validation_result = validate_iam_policy(policy)
if validation_result.is_valid:
    print("Policy is valid")
else:
    print(f"Policy errors: {validation_result.errors}")
```

### Audit and Monitoring
Regular policy review and monitoring:

```python
from pynomaly.security import audit_iam_permissions

# Generate permission audit report
audit_report = audit_iam_permissions(
    principal="arn:aws:iam::123456789012:user/data-scientist",
    timeframe_days=30
)

print(f"Unused permissions: {audit_report.unused_permissions}")
print(f"Excessive permissions: {audit_report.excessive_permissions}")
```

## Security Hardening Checklist

- [ ] **MFA Required**: Enable multi-factor authentication for all privileged access
- [ ] **IP Restrictions**: Limit access to known IP ranges
- [ ] **Time Bounds**: Set expiration dates for temporary access
- [ ] **Resource Specific**: Use specific ARNs instead of wildcards
- [ ] **Audit Logging**: Enable comprehensive audit trails
- [ ] **Regular Review**: Schedule periodic permission reviews
- [ ] **Separation of Duties**: Implement role separation between environments
- [ ] **Encryption**: Require encryption in transit and at rest
- [ ] **Rate Limiting**: Implement API rate limiting
- [ ] **Monitoring**: Set up alerts for unusual access patterns

## Related Documentation

- [Security Best Practices](security-best-practices.md)
- [Authentication Guide](authentication.md)
- [Encryption Configuration](encryption.md)
- [Audit Logging](audit-logging.md)
- [Deployment Security](../deployment/SECURITY.md)

---

📍 **Location**: `docs/security/`  
🏠 **Documentation Home**: [docs/](../README.md)  
🔗 **Security Home**: [Security Guide](README.md)
