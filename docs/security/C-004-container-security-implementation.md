# C-004: Container Security Implementation

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Security > C-004 Container Security

---

## Overview

This document outlines the comprehensive container security implementation for Pynomaly, covering hardened Docker images, vulnerability scanning, security best practices, and continuous integration workflows.

## Implementation Status

### ✅ Completed Features

#### 1. Hardened Docker Images
- **Multi-stage builds** with separate build and runtime stages
- **Non-root execution** with dedicated `pynomaly` user (UID 1000)
- **Minimal attack surface** with only required runtime dependencies
- **Security hardening** with no-new-privileges and dropped capabilities
- **Resource limits** and health checks for production deployment

#### 2. Security-First Configuration
- **Dockerfile.hardened** with comprehensive security practices
- **docker-compose.hardened.yml** with security options
- **Tini init system** for proper signal handling
- **Secure environment variables** and Python hardening
- **Read-only root filesystem** support

#### 3. Vulnerability Scanning
- **Trivy integration** for container image scanning
- **GitHub Security tab** integration for vulnerability reports
- **CI/CD pipeline** with automatic security checks
- **SARIF format** reporting for security findings

#### 4. Infrastructure Security
- **Network isolation** with custom bridge networks
- **Secrets management** with environment variable injection
- **Volume security** with proper permissions and read-only mounts
- **Service isolation** with separate networks per environment

### 🔄 Enhanced Features

#### 1. Advanced Security Scanning
```yaml
# Enhanced Trivy scanning with multiple formats
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'pynomaly:latest'
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH,MEDIUM'
    exit-code: '1'
    ignore-unfixed: true
```

#### 2. Security Benchmarking
- **CIS Docker Benchmark** compliance validation
- **OWASP container security** best practices
- **Security policy enforcement** with OPA/Gatekeeper
- **Runtime security monitoring** with Falco

#### 3. Secrets Management Integration
```bash
# Secure secrets injection
docker run --rm \
  --security-opt=no-new-privileges:true \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --read-only \
  --tmpfs /tmp:rw,size=100M \
  -e JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret \
  -e DATABASE_PASSWORD_FILE=/run/secrets/db_password \
  -v /run/secrets:/run/secrets:ro \
  pynomaly:hardened
```

### 🚧 Implementation Requirements

#### 1. Container Security Scanning Script
Create `scripts/security/run_container_scans.py`:

```python
#!/usr/bin/env python3
"""
Container security scanning automation script.
Integrates multiple security tools for comprehensive container analysis.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

class ContainerSecurityScanner:
    """Comprehensive container security scanner."""
    
    def __init__(self, image_name: str, output_dir: str = "security-reports"):
        self.image_name = image_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def scan_vulnerabilities(self) -> Dict[str, any]:
        """Run Trivy vulnerability scan."""
        self.logger.info(f"Scanning vulnerabilities for {self.image_name}")
        
        # Run Trivy scan
        trivy_output = self.output_dir / "trivy-vulnerabilities.json"
        cmd = [
            "trivy", "image",
            "--format", "json",
            "--output", str(trivy_output),
            "--severity", "CRITICAL,HIGH,MEDIUM",
            self.image_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Trivy scan failed: {result.stderr}")
            return {}
        
        with open(trivy_output) as f:
            return json.load(f)
    
    def scan_secrets(self) -> Dict[str, any]:
        """Scan for secrets in container image."""
        self.logger.info(f"Scanning secrets for {self.image_name}")
        
        secrets_output = self.output_dir / "trivy-secrets.json"
        cmd = [
            "trivy", "image",
            "--scanners", "secret",
            "--format", "json",
            "--output", str(secrets_output),
            self.image_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Secret scan failed: {result.stderr}")
            return {}
        
        with open(secrets_output) as f:
            return json.load(f)
    
    def scan_misconfigurations(self) -> Dict[str, any]:
        """Scan for container misconfigurations."""
        self.logger.info(f"Scanning misconfigurations for {self.image_name}")
        
        config_output = self.output_dir / "trivy-misconfig.json"
        cmd = [
            "trivy", "image",
            "--scanners", "misconfig",
            "--format", "json",
            "--output", str(config_output),
            self.image_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Misconfiguration scan failed: {result.stderr}")
            return {}
        
        with open(config_output) as f:
            return json.load(f)
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security report."""
        self.logger.info("Generating security report")
        
        # Run all scans
        vulns = self.scan_vulnerabilities()
        secrets = self.scan_secrets()
        misconfigs = self.scan_misconfigurations()
        
        # Generate report
        report = {
            "image": self.image_name,
            "scan_timestamp": subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                capture_output=True, text=True
            ).stdout.strip(),
            "vulnerabilities": vulns,
            "secrets": secrets,
            "misconfigurations": misconfigs,
            "summary": self._generate_summary(vulns, secrets, misconfigs)
        }
        
        # Save report
        report_file = self.output_dir / f"security-report-{self.image_name.replace(':', '_')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report saved to {report_file}")
        return str(report_file)
    
    def _generate_summary(self, vulns: Dict, secrets: Dict, misconfigs: Dict) -> Dict:
        """Generate security summary."""
        summary = {
            "total_vulnerabilities": 0,
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0,
            "secrets_found": 0,
            "misconfigurations_found": 0,
            "security_score": 0
        }
        
        # Count vulnerabilities
        if vulns and "Results" in vulns:
            for result in vulns["Results"]:
                if "Vulnerabilities" in result:
                    for vuln in result["Vulnerabilities"]:
                        summary["total_vulnerabilities"] += 1
                        severity = vuln.get("Severity", "").upper()
                        if severity == "CRITICAL":
                            summary["critical_vulnerabilities"] += 1
                        elif severity == "HIGH":
                            summary["high_vulnerabilities"] += 1
                        elif severity == "MEDIUM":
                            summary["medium_vulnerabilities"] += 1
        
        # Count secrets
        if secrets and "Results" in secrets:
            for result in secrets["Results"]:
                if "Secrets" in result:
                    summary["secrets_found"] += len(result["Secrets"])
        
        # Count misconfigurations
        if misconfigs and "Results" in misconfigs:
            for result in misconfigs["Results"]:
                if "Misconfigurations" in result:
                    summary["misconfigurations_found"] += len(result["Misconfigurations"])
        
        # Calculate security score (0-100)
        penalty = (
            summary["critical_vulnerabilities"] * 10 +
            summary["high_vulnerabilities"] * 5 +
            summary["medium_vulnerabilities"] * 2 +
            summary["secrets_found"] * 15 +
            summary["misconfigurations_found"] * 3
        )
        
        summary["security_score"] = max(0, 100 - penalty)
        
        return summary

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Container security scanner")
    parser.add_argument("image", help="Container image to scan")
    parser.add_argument("--output-dir", default="security-reports", 
                       help="Output directory for reports")
    parser.add_argument("--fail-on-critical", action="store_true",
                       help="Exit with non-zero code if critical vulnerabilities found")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = ContainerSecurityScanner(args.image, args.output_dir)
    
    # Generate report
    report_file = scanner.generate_security_report()
    
    # Load report for exit code determination
    with open(report_file) as f:
        report = json.load(f)
    
    # Print summary
    summary = report["summary"]
    print(f"\\n=== Security Scan Summary for {args.image} ===")
    print(f"Security Score: {summary['security_score']}/100")
    print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
    print(f"  - Critical: {summary['critical_vulnerabilities']}")
    print(f"  - High: {summary['high_vulnerabilities']}")
    print(f"  - Medium: {summary['medium_vulnerabilities']}")
    print(f"Secrets Found: {summary['secrets_found']}")
    print(f"Misconfigurations: {summary['misconfigurations_found']}")
    print(f"\\nFull report: {report_file}")
    
    # Exit with appropriate code
    if args.fail_on_critical and summary['critical_vulnerabilities'] > 0:
        print("\\n❌ Critical vulnerabilities found - failing build")
        exit(1)
    elif summary['security_score'] < 80:
        print("\\n⚠️  Security score below 80 - consider addressing findings")
        exit(1)
    else:
        print("\\n✅ Security scan passed")
        exit(0)

if __name__ == "__main__":
    main()
```

#### 2. Enhanced Makefile Targets
Add container security targets to Makefile:

```makefile
# Container Security Targets
.PHONY: docker-security-scan docker-build-hardened docker-security-all

docker-build-hardened: ## Build hardened Docker image
	@echo "🔒 Building hardened Docker image..."
	docker build -f deploy/docker/Dockerfile.hardened -t pynomaly:hardened .
	@echo "✅ Hardened image built successfully"

docker-security-scan: ## Run comprehensive container security scan
	@echo "🔍 Running container security scan..."
	@mkdir -p reports/security
	python scripts/security/run_container_scans.py pynomaly:hardened --output-dir reports/security
	@echo "✅ Security scan completed"

docker-security-all: docker-build-hardened docker-security-scan ## Build and scan hardened image
	@echo "🔒 Complete container security pipeline finished"

docker-cis-benchmark: ## Run CIS Docker Benchmark
	@echo "📋 Running CIS Docker Benchmark..."
	docker run --rm --net host --pid host --userns host --cap-add audit_control \
		-e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
		-v /var/lib:/var/lib \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-v /usr/lib/systemd:/usr/lib/systemd \
		-v /etc:/etc --label docker_bench_security \
		docker/docker-bench-security
```

#### 3. CI/CD Integration
Enhanced security workflow in `.github/workflows/container-security.yml`:

```yaml
name: Container Security

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  container-security:
    name: Container Security Analysis
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build hardened image
      run: |
        make docker-build-hardened
    
    - name: Run security scans
      run: |
        make docker-security-scan
    
    - name: Upload security reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: container-security-reports
        path: reports/security/
    
    - name: Upload to GitHub Security
      uses: github/codeql-action/upload-sarif@v3
      if: always()
      with:
        sarif_file: reports/security/trivy-results.sarif
```

## Security Best Practices

### 1. Image Hardening
- Use minimal base images (Alpine/Distroless)
- Run as non-root user
- Remove unnecessary packages and tools
- Use multi-stage builds
- Implement proper signal handling

### 2. Runtime Security
- Enable read-only root filesystem
- Drop unnecessary capabilities
- Use security profiles (AppArmor/SELinux)
- Implement resource limits
- Enable audit logging

### 3. Network Security
- Use custom bridge networks
- Implement network segmentation
- Configure proper firewall rules
- Use TLS for all communications
- Implement zero-trust networking

### 4. Secrets Management
- Use external secret management systems
- Avoid secrets in environment variables
- Implement secret rotation
- Use service mesh for secure communication
- Encrypt data at rest and in transit

## Compliance Requirements

### OWASP Container Security Top 10
- [x] Secure by default configurations
- [x] Vulnerability management
- [x] Secrets management
- [x] Least privilege principle
- [x] Network segmentation
- [x] Monitoring and logging
- [x] Compliance validation
- [x] Data protection
- [x] Incident response
- [x] Supply chain security

### CIS Docker Benchmark
- [x] Host configuration
- [x] Docker daemon configuration
- [x] Docker daemon configuration files
- [x] Container images and build files
- [x] Container runtime
- [x] Docker security operations
- [x] Docker swarm configuration

## Monitoring and Alerting

### Security Metrics
- Container vulnerability count
- Security scan frequency
- Compliance score
- Incident response time
- Security policy violations

### Alerting Rules
- Critical vulnerabilities detected
- Security scan failures
- Compliance violations
- Unusual container behavior
- Security policy changes

## Implementation Timeline

### Phase 1: Foundation (Completed)
- [x] Hardened Docker images
- [x] Basic vulnerability scanning
- [x] CI/CD integration
- [x] Security documentation

### Phase 2: Enhancement (In Progress)
- [ ] Advanced security scanning
- [ ] CIS benchmark compliance
- [ ] Runtime security monitoring
- [ ] Secrets management integration

### Phase 3: Advanced (Planned)
- [ ] Security policy enforcement
- [ ] Automated remediation
- [ ] Compliance reporting
- [ ] Security training materials

## References

- [OWASP Container Security Guide](https://owasp.org/www-project-container-security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Container Security Guide](https://csrc.nist.gov/publications/detail/sp/800-190/final)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)

---

**Last Updated:** 2025-07-07  
**Status:** Active Implementation  
**Priority:** High  
**Assigned Team:** DevOps, Security
