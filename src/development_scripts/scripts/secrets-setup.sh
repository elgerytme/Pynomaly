#!/bin/bash

# Pynomaly Production Secrets Setup Script
# This script sets up AWS Secrets Manager and Kubernetes secrets for production deployment

set -euo pipefail

# Configuration
AWS_REGION="${AWS_REGION:-us-west-2}"
NAMESPACE="${NAMESPACE:-pynomaly-production}"
SECRET_PREFIX="pynomaly/production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if required tools are installed
    for tool in aws kubectl jq openssl; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi

    # Check kubectl context
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl not configured or cluster not accessible"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Generate secure random passwords and keys
generate_secrets() {
    log_info "Generating secure secrets..."

    # Database secrets
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    POSTGRES_USER="pynomaly_user"
    POSTGRES_DB="pynomaly_prod"

    # Redis secrets
    REDIS_PASSWORD=$(openssl rand -base64 32)

    # Application secrets
    APP_SECRET_KEY=$(openssl rand -base64 64)
    JWT_SECRET_KEY=$(openssl rand -base64 64)
    JWT_REFRESH_SECRET_KEY=$(openssl rand -base64 64)
    ENCRYPTION_KEY=$(openssl rand -base64 32)

    # Monitoring secrets
    PROMETHEUS_PASSWORD=$(openssl rand -base64 24)
    GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 24)
    MONITORING_API_KEY=$(openssl rand -base64 32)

    log_success "Secrets generated successfully"
}

# Create AWS Secrets Manager secrets
create_aws_secrets() {
    log_info "Creating AWS Secrets Manager secrets..."

    # Database secrets
    log_info "Creating database secrets..."
    aws secretsmanager create-secret \
        --name "${SECRET_PREFIX}/database" \
        --description "Pynomaly production database credentials" \
        --secret-string "{
            \"username\": \"$POSTGRES_USER\",
            \"password\": \"$POSTGRES_PASSWORD\",
            \"database\": \"$POSTGRES_DB\",
            \"host\": \"postgres-service.${NAMESPACE}.svc.cluster.local\",
            \"port\": \"5432\"
        }" \
        --region $AWS_REGION \
        --tags '[
            {"Key": "Environment", "Value": "production"},
            {"Key": "Application", "Value": "pynomaly"},
            {"Key": "Component", "Value": "database"}
        ]' 2>/dev/null || {
            log_warning "Database secret already exists, updating..."
            aws secretsmanager put-secret-value \
                --secret-id "${SECRET_PREFIX}/database" \
                --secret-string "{
                    \"username\": \"$POSTGRES_USER\",
                    \"password\": \"$POSTGRES_PASSWORD\",
                    \"database\": \"$POSTGRES_DB\",
                    \"host\": \"postgres-service.${NAMESPACE}.svc.cluster.local\",
                    \"port\": \"5432\"
                }" \
                --region $AWS_REGION
        }

    # Redis secrets
    log_info "Creating Redis secrets..."
    aws secretsmanager create-secret \
        --name "${SECRET_PREFIX}/redis" \
        --description "Pynomaly production Redis credentials" \
        --secret-string "{
            \"password\": \"$REDIS_PASSWORD\",
            \"host\": \"redis-service.${NAMESPACE}.svc.cluster.local\",
            \"port\": \"6379\"
        }" \
        --region $AWS_REGION \
        --tags '[
            {"Key": "Environment", "Value": "production"},
            {"Key": "Application", "Value": "pynomaly"},
            {"Key": "Component", "Value": "redis"}
        ]' 2>/dev/null || {
            log_warning "Redis secret already exists, updating..."
            aws secretsmanager put-secret-value \
                --secret-id "${SECRET_PREFIX}/redis" \
                --secret-string "{
                    \"password\": \"$REDIS_PASSWORD\",
                    \"host\": \"redis-service.${NAMESPACE}.svc.cluster.local\",
                    \"port\": \"6379\"
                }" \
                --region $AWS_REGION
        }

    # Application secrets
    log_info "Creating application secrets..."
    aws secretsmanager create-secret \
        --name "${SECRET_PREFIX}/app" \
        --description "Pynomaly production application secrets" \
        --secret-string "{
            \"secret_key\": \"$APP_SECRET_KEY\",
            \"jwt_secret_key\": \"$JWT_SECRET_KEY\",
            \"jwt_refresh_secret_key\": \"$JWT_REFRESH_SECRET_KEY\",
            \"encryption_key\": \"$ENCRYPTION_KEY\"
        }" \
        --region $AWS_REGION \
        --tags '[
            {"Key": "Environment", "Value": "production"},
            {"Key": "Application", "Value": "pynomaly"},
            {"Key": "Component", "Value": "application"}
        ]' 2>/dev/null || {
            log_warning "Application secret already exists, updating..."
            aws secretsmanager put-secret-value \
                --secret-id "${SECRET_PREFIX}/app" \
                --secret-string "{
                    \"secret_key\": \"$APP_SECRET_KEY\",
                    \"jwt_secret_key\": \"$JWT_SECRET_KEY\",
                    \"jwt_refresh_secret_key\": \"$JWT_REFRESH_SECRET_KEY\",
                    \"encryption_key\": \"$ENCRYPTION_KEY\"
                }" \
                --region $AWS_REGION
        }

    # Monitoring secrets
    log_info "Creating monitoring secrets..."
    aws secretsmanager create-secret \
        --name "${SECRET_PREFIX}/monitoring" \
        --description "Pynomaly production monitoring secrets" \
        --secret-string "{
            \"prometheus_password\": \"$PROMETHEUS_PASSWORD\",
            \"grafana_admin_password\": \"$GRAFANA_ADMIN_PASSWORD\",
            \"api_key\": \"$MONITORING_API_KEY\"
        }" \
        --region $AWS_REGION \
        --tags '[
            {"Key": "Environment", "Value": "production"},
            {"Key": "Application", "Value": "pynomaly"},
            {"Key": "Component", "Value": "monitoring"}
        ]' 2>/dev/null || {
            log_warning "Monitoring secret already exists, updating..."
            aws secretsmanager put-secret-value \
                --secret-id "${SECRET_PREFIX}/monitoring" \
                --secret-string "{
                    \"prometheus_password\": \"$PROMETHEUS_PASSWORD\",
                    \"grafana_admin_password\": \"$GRAFANA_ADMIN_PASSWORD\",
                    \"api_key\": \"$MONITORING_API_KEY\"
                }" \
                --region $AWS_REGION
        }

    log_success "AWS Secrets Manager secrets created/updated"
}

# Create Kubernetes namespace and RBAC
setup_kubernetes() {
    log_info "Setting up Kubernetes namespace and RBAC..."

    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

    # Label the namespace
    kubectl label namespace $NAMESPACE \
        environment=production \
        app=pynomaly \
        secrets-management=enabled \
        --overwrite

    log_success "Kubernetes namespace configured"
}

# Install External Secrets Operator if not present
install_external_secrets_operator() {
    log_info "Checking External Secrets Operator installation..."

    if ! kubectl get crd externalsecrets.external-secrets.io &> /dev/null; then
        log_info "Installing External Secrets Operator..."

        helm repo add external-secrets https://charts.external-secrets.io
        helm repo update

        helm install external-secrets external-secrets/external-secrets \
            --namespace external-secrets-system \
            --create-namespace \
            --set installCRDs=true \
            --set webhook.port=9443 \
            --set certController.create=true

        # Wait for the operator to be ready
        kubectl wait --for=condition=ready pod \
            --selector=app.kubernetes.io/name=external-secrets \
            --namespace=external-secrets-system \
            --timeout=300s

        log_success "External Secrets Operator installed"
    else
        log_info "External Secrets Operator already installed"
    fi
}

# Create IAM role for external secrets (if using EKS)
create_iam_role() {
    local AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    local OIDC_ISSUER=$(aws eks describe-cluster --name $(kubectl config current-context | cut -d'/' -f2) --query "cluster.identity.oidc.issuer" --output text 2>/dev/null || echo "")

    if [[ -n "$OIDC_ISSUER" ]]; then
        log_info "Creating IAM role for External Secrets..."

        # Trust policy for OIDC
        cat > /tmp/trust-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_ISSUER#https://}"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "${OIDC_ISSUER#https://}:sub": "system:serviceaccount:${NAMESPACE}:pynomaly-secrets-sa",
                    "${OIDC_ISSUER#https://}:aud": "sts.amazonaws.com"
                }
            }
        }
    ]
}
EOF

        # Create IAM role
        aws iam create-role \
            --role-name pynomaly-secrets-role \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --description "Role for Pynomaly External Secrets access" 2>/dev/null || {
                log_warning "IAM role already exists, updating trust policy..."
                aws iam update-assume-role-policy \
                    --role-name pynomaly-secrets-role \
                    --policy-document file:///tmp/trust-policy.json
            }

        # Attach Secrets Manager policy
        cat > /tmp/secrets-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:DescribeSecret"
            ],
            "Resource": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${SECRET_PREFIX}/*"
        }
    ]
}
EOF

        aws iam put-role-policy \
            --role-name pynomaly-secrets-role \
            --policy-name SecretsManagerAccess \
            --policy-document file:///tmp/secrets-policy.json

        # Clean up temp files
        rm -f /tmp/trust-policy.json /tmp/secrets-policy.json

        log_success "IAM role created/updated for External Secrets"
    else
        log_warning "Not running on EKS, skipping IAM role creation"
    fi
}

# Apply Kubernetes secret manifests
apply_secret_manifests() {
    log_info "Applying Kubernetes secret manifests..."

    # Apply the secrets management configuration
    kubectl apply -f k8s/production/secrets-management.yaml

    # Wait for External Secrets to sync
    log_info "Waiting for External Secrets to sync..."
    kubectl wait --for=condition=Ready externalsecret \
        pynomaly-external-secrets \
        --namespace=$NAMESPACE \
        --timeout=300s 2>/dev/null || {
            log_warning "External Secret sync timeout, checking manually..."
            kubectl get externalsecret pynomaly-external-secrets -n $NAMESPACE -o yaml
        }

    log_success "Kubernetes secrets configured"
}

# Verify secrets are properly created
verify_secrets() {
    log_info "Verifying secrets configuration..."

    # Check AWS Secrets Manager
    for secret in database redis app monitoring; do
        if aws secretsmanager describe-secret --secret-id "${SECRET_PREFIX}/${secret}" --region $AWS_REGION &> /dev/null; then
            log_success "AWS secret ${SECRET_PREFIX}/${secret} exists"
        else
            log_error "AWS secret ${SECRET_PREFIX}/${secret} not found"
        fi
    done

    # Check Kubernetes secrets
    if kubectl get secret pynomaly-production-secrets -n $NAMESPACE &> /dev/null; then
        log_success "Kubernetes secret pynomaly-production-secrets exists"

        # Show secret keys (not values)
        log_info "Secret keys available:"
        kubectl get secret pynomaly-production-secrets -n $NAMESPACE -o jsonpath='{.data}' | jq -r 'keys[]' | while read key; do
            echo "  - $key"
        done
    else
        log_error "Kubernetes secret pynomaly-production-secrets not found"
    fi

    log_success "Secrets verification completed"
}

# Generate documentation
generate_documentation() {
    log_info "Generating secrets documentation..."

    cat > docs/secrets-management.md <<EOF
# Pynomaly Production Secrets Management

## Overview
This document describes the secrets management setup for Pynomaly production deployment.

## Architecture
- **AWS Secrets Manager**: Centralized secret storage
- **External Secrets Operator**: Kubernetes integration
- **Automatic Rotation**: Weekly secret rotation via CronJob

## Secrets Structure

### Database Secrets (\`${SECRET_PREFIX}/database\`)
- username: Database username
- password: Database password
- database: Database name
- host: Database host
- port: Database port

### Redis Secrets (\`${SECRET_PREFIX}/redis\`)
- password: Redis password
- host: Redis host
- port: Redis port

### Application Secrets (\`${SECRET_PREFIX}/app\`)
- secret_key: Application secret key
- jwt_secret_key: JWT signing key
- jwt_refresh_secret_key: JWT refresh token key
- encryption_key: Data encryption key

### Monitoring Secrets (\`${SECRET_PREFIX}/monitoring\`)
- prometheus_password: Prometheus admin password
- grafana_admin_password: Grafana admin password
- api_key: Monitoring API key

## Secret Rotation
Secrets are automatically rotated weekly using a Kubernetes CronJob.
Manual rotation can be triggered by running the secret rotation job.

## Access Control
- Kubernetes RBAC limits secret access to authorized service accounts
- AWS IAM policies restrict Secrets Manager access
- Network policies control egress to AWS Secrets Manager

## Emergency Procedures

### Manual Secret Rotation
\`\`\`bash
kubectl create job --from=cronjob/pynomaly-secret-rotation manual-rotation-\$(date +%Y%m%d) -n ${NAMESPACE}
\`\`\`

### Secret Recovery
1. Check AWS Secrets Manager console
2. Verify External Secrets status: \`kubectl get externalsecret -n ${NAMESPACE}\`
3. Force resync: \`kubectl annotate externalsecret pynomaly-external-secrets force-sync=\$(date +%s) -n ${NAMESPACE}\`

### Troubleshooting
- Check External Secrets logs: \`kubectl logs -l app.kubernetes.io/name=external-secrets -n external-secrets-system\`
- Verify IAM permissions for the service account
- Check AWS Secrets Manager access logs

## Security Considerations
- Secrets are encrypted in transit and at rest
- Access is logged and monitored
- Regular rotation minimizes exposure risk
- Principle of least privilege is enforced

Generated on: \$(date)
EOF

    log_success "Documentation generated: docs/secrets-management.md"
}

# Main execution
main() {
    log_info "Starting Pynomaly production secrets setup..."

    # Create docs directory if it doesn't exist
    mkdir -p docs

    check_prerequisites
    generate_secrets
    create_aws_secrets
    setup_kubernetes
    install_external_secrets_operator
    create_iam_role
    apply_secret_manifests
    verify_secrets
    generate_documentation

    log_success "Pynomaly production secrets setup completed successfully!"
    log_info "Next steps:"
    echo "  1. Review the generated documentation in docs/secrets-management.md"
    echo "  2. Test the application deployment with the new secrets"
    echo "  3. Monitor the External Secrets Operator for any sync issues"
    echo "  4. Set up monitoring alerts for secret rotation failures"
}

# Execute main function
main "$@"
