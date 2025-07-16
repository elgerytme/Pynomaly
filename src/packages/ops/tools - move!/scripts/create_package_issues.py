#!/usr/bin/env python3
"""
Script to create comprehensive GitHub issues for the 10-package domain expansion.
Uses the GitHub CLI (gh) to create issues with proper labeling and structure.
"""

import subprocess
import json
from typing import Dict, List

# Package definitions
PACKAGES = {
    "domain_library": {
        "name": "Domain Library",
        "description": "Domain Catalog & Logic Management",
        "epic_issue": "217"
    },
    "data_architecture": {
        "name": "Data Architecture", 
        "description": "Data Architecture Governance",
        "epic_issue": "218"
    },
    "data_engineering": {
        "name": "Data Engineering",
        "description": "Data Pipeline Engineering", 
        "epic_issue": "220"
    },
    "software_architecture": {
        "name": "Software Architecture",
        "description": "Software Architecture Management",
        "epic_issue": "221"
    },
    "software_engineering": {
        "name": "Software Engineering", 
        "description": "Software Engineering Tools",
        "epic_issue": "223"
    },
    "testing": {
        "name": "Testing Framework",
        "description": "Advanced Testing Framework",
        "epic_issue": "224"
    },
    "knowledge_graph": {
        "name": "Knowledge Graph",
        "description": "Knowledge Graph System", 
        "epic_issue": "226"
    },
    "aiops": {
        "name": "AIOps",
        "description": "AIOps Platform",
        "epic_issue": "228"
    },
    "logic": {
        "name": "Logic",
        "description": "Formal Logic Tools",
        "epic_issue": "229"
    },
    "category_theory": {
        "name": "Category Theory",
        "description": "Category Theory Applications",
        "epic_issue": "230"
    }
}

# Issue templates by category
ISSUE_TEMPLATES = {
    "requirements": [
        {
            "code": "REQ-001", 
            "title": "Create Business Requirements Document",
            "description": "Create comprehensive BRD defining scope, functionality, and business value",
            "priority": "high"
        },
        {
            "code": "REQ-002",
            "title": "Define Domain Entities and Models", 
            "description": "Define core domain entities, value objects, and data models",
            "priority": "high"
        },
        {
            "code": "REQ-003",
            "title": "Create Use Cases and User Stories",
            "description": "Document detailed use cases and user stories with acceptance criteria",
            "priority": "high"
        },
        {
            "code": "REQ-004", 
            "title": "Design API Specifications",
            "description": "Create OpenAPI specifications for all REST endpoints",
            "priority": "high"
        },
        {
            "code": "REQ-005",
            "title": "Create Architecture Decision Records",
            "description": "Document key architectural decisions and trade-offs",
            "priority": "medium"
        },
        {
            "code": "REQ-006",
            "title": "Define Data Models and Schemas",
            "description": "Design database schemas and data validation models",
            "priority": "high"
        },
        {
            "code": "REQ-007",
            "title": "Create BDD Scenarios and Story Maps",
            "description": "Develop behavior-driven development scenarios and user story maps",
            "priority": "medium"
        },
        {
            "code": "REQ-008",
            "title": "Define Security Requirements",
            "description": "Specify authentication, authorization, and security requirements",
            "priority": "high"
        },
        {
            "code": "REQ-009", 
            "title": "Define Performance Requirements",
            "description": "Establish performance benchmarks and scalability requirements",
            "priority": "medium"
        },
        {
            "code": "REQ-010",
            "title": "Create Integration Requirements", 
            "description": "Define integration points with existing Pynomaly ecosystem",
            "priority": "high"
        }
    ],
    "domain": [
        {
            "code": "DOM-001",
            "title": "Implement Core Domain Entities",
            "description": "Implement main domain entities following DDD patterns",
            "priority": "high"
        },
        {
            "code": "DOM-002", 
            "title": "Create Value Objects",
            "description": "Implement value objects with validation and immutability",
            "priority": "high"
        },
        {
            "code": "DOM-003",
            "title": "Define Domain Services",
            "description": "Implement domain services for business logic operations",
            "priority": "high"
        },
        {
            "code": "DOM-004",
            "title": "Implement Repository Abstractions",
            "description": "Create repository interfaces for data access abstraction",
            "priority": "high"
        },
        {
            "code": "DOM-005",
            "title": "Create Domain Events",
            "description": "Implement domain events for loose coupling and integration",
            "priority": "medium"
        },
        {
            "code": "DOM-006",
            "title": "Implement Business Rules Validation",
            "description": "Create business rule validation and enforcement",
            "priority": "high"
        },
        {
            "code": "DOM-007",
            "title": "Create Domain Exceptions",
            "description": "Implement domain-specific exception handling",
            "priority": "medium"
        },
        {
            "code": "DOM-008",
            "title": "Implement Aggregate Roots",
            "description": "Create aggregate root entities with consistency boundaries",
            "priority": "high"
        },
        {
            "code": "DOM-009",
            "title": "Create Domain Specifications",
            "description": "Implement specification pattern for complex queries",
            "priority": "medium"
        },
        {
            "code": "DOM-010",
            "title": "Implement Domain Factories",
            "description": "Create factory patterns for complex object creation",
            "priority": "medium"
        },
        {
            "code": "DOM-011",
            "title": "Create Domain Policies",
            "description": "Implement business policies and rules engine",
            "priority": "medium"
        },
        {
            "code": "DOM-012",
            "title": "Implement Domain Workflows",
            "description": "Create domain workflow orchestration",
            "priority": "medium"
        },
        {
            "code": "DOM-013",
            "title": "Create Domain Invariants",
            "description": "Implement domain invariant validation",
            "priority": "medium"
        },
        {
            "code": "DOM-014",
            "title": "Implement Domain Calculations",
            "description": "Create domain calculation and computation logic",
            "priority": "medium"
        },
        {
            "code": "DOM-015",
            "title": "Create Domain Validation Rules",
            "description": "Implement comprehensive domain validation framework",
            "priority": "medium"
        }
    ],
    "application": [
        {
            "code": "APP-001",
            "title": "Create Application Services", 
            "description": "Implement application orchestration services",
            "priority": "high"
        },
        {
            "code": "APP-002",
            "title": "Implement Use Case Handlers",
            "description": "Create handlers for all documented use cases",
            "priority": "high"
        },
        {
            "code": "APP-003",
            "title": "Create DTOs and Mappers",
            "description": "Implement data transfer objects and mapping logic",
            "priority": "high"
        },
        {
            "code": "APP-004",
            "title": "Implement Command Handlers",
            "description": "Create CQRS command handlers for write operations",
            "priority": "medium"
        },
        {
            "code": "APP-005",
            "title": "Create Query Handlers", 
            "description": "Implement query handlers for read operations",
            "priority": "medium"
        },
        {
            "code": "APP-006",
            "title": "Implement Event Handlers",
            "description": "Create domain event handlers for integration",
            "priority": "medium"
        },
        {
            "code": "APP-007",
            "title": "Create Application Workflows",
            "description": "Implement complex business workflows",
            "priority": "medium"
        },
        {
            "code": "APP-008",
            "title": "Implement Cross-Cutting Concerns",
            "description": "Add logging, caching, and monitoring",
            "priority": "medium"
        },
        {
            "code": "APP-009",
            "title": "Create Application Validators",
            "description": "Implement application-level validation",
            "priority": "medium"
        },
        {
            "code": "APP-010",
            "title": "Implement Application Security",
            "description": "Add authorization and security controls",
            "priority": "high"
        },
        {
            "code": "APP-011",
            "title": "Create Application Monitoring",
            "description": "Implement application metrics and monitoring",
            "priority": "medium"
        },
        {
            "code": "APP-012",
            "title": "Implement Application Caching",
            "description": "Add caching strategies and implementation",
            "priority": "medium"
        },
        {
            "code": "APP-013",
            "title": "Create Application Logging",
            "description": "Implement structured logging framework",
            "priority": "medium"
        },
        {
            "code": "APP-014",
            "title": "Implement Application Configuration",
            "description": "Add configuration management and validation",
            "priority": "medium"
        },
        {
            "code": "APP-015",
            "title": "Create Application Health Checks",
            "description": "Implement health monitoring and diagnostics",
            "priority": "medium"
        }
    ],
    "infrastructure": [
        {
            "code": "INF-001",
            "title": "Implement Persistence Layer",
            "description": "Create database repositories and data access implementation",
            "priority": "high"
        },
        {
            "code": "INF-002",
            "title": "Create Database Adapters",
            "description": "Implement adapters for different database technologies",
            "priority": "high"
        },
        {
            "code": "INF-003",
            "title": "Implement External Service Adapters",
            "description": "Create adapters for external API integrations",
            "priority": "medium"
        },
        {
            "code": "INF-004",
            "title": "Create Monitoring Infrastructure",
            "description": "Implement logging, metrics, and observability",
            "priority": "medium"
        },
        {
            "code": "INF-005",
            "title": "Implement Security Infrastructure",
            "description": "Create authentication, authorization, and security components",
            "priority": "high"
        },
        {
            "code": "INF-006",
            "title": "Create Caching Infrastructure",
            "description": "Implement Redis and in-memory caching systems",
            "priority": "medium"
        },
        {
            "code": "INF-007",
            "title": "Implement Messaging Infrastructure",
            "description": "Create event bus and message queue systems",
            "priority": "medium"
        },
        {
            "code": "INF-008",
            "title": "Create Logging Infrastructure",
            "description": "Implement structured logging and log aggregation",
            "priority": "medium"
        },
        {
            "code": "INF-009",
            "title": "Implement Configuration Management",
            "description": "Create configuration loading and validation",
            "priority": "medium"
        },
        {
            "code": "INF-010",
            "title": "Create Deployment Infrastructure",
            "description": "Implement Docker, Kubernetes, and CI/CD support",
            "priority": "medium"
        }
    ],
    "presentation": [
        {
            "code": "API-001",
            "title": "Create FastAPI Application Structure",
            "description": "Set up FastAPI application with proper structure",
            "priority": "high"
        },
        {
            "code": "API-002", 
            "title": "Implement REST API Endpoints",
            "description": "Create all REST API endpoints with validation",
            "priority": "high"
        },
        {
            "code": "API-003",
            "title": "Create API Authentication/Authorization",
            "description": "Implement JWT and OAuth2 authentication",
            "priority": "high"
        },
        {
            "code": "API-004",
            "title": "Implement API Validation",
            "description": "Add request/response validation with Pydantic",
            "priority": "high"
        },
        {
            "code": "API-005",
            "title": "Create API Documentation",
            "description": "Generate comprehensive OpenAPI documentation",
            "priority": "medium"
        },
        {
            "code": "CLI-001",
            "title": "Create Typer CLI Application",
            "description": "Implement command-line interface using Typer",
            "priority": "high"
        },
        {
            "code": "CLI-002",
            "title": "Implement CLI Commands",
            "description": "Create all CLI commands with proper help",
            "priority": "high"
        },
        {
            "code": "CLI-003",
            "title": "Create CLI Help and Documentation",
            "description": "Add comprehensive CLI documentation and examples",
            "priority": "medium"
        },
        {
            "code": "WEB-001",
            "title": "Create React Web UI Components",
            "description": "Build responsive web user interface",
            "priority": "medium"
        },
        {
            "code": "WEB-002",
            "title": "Implement Web UI Forms and Validation",
            "description": "Create forms with client-side validation",
            "priority": "medium"
        },
        {
            "code": "WEB-003",
            "title": "Create Web UI Routing and Navigation",
            "description": "Implement client-side routing and navigation",
            "priority": "medium"
        },
        {
            "code": "SDK-001",
            "title": "Generate Python SDK",
            "description": "Create Python SDK for programmatic access",
            "priority": "medium"
        },
        {
            "code": "SDK-002",
            "title": "Generate TypeScript SDK",
            "description": "Create TypeScript SDK for web integrations",
            "priority": "medium"
        },
        {
            "code": "SDK-003",
            "title": "Create SDK Documentation",
            "description": "Generate comprehensive SDK documentation",
            "priority": "medium"
        },
        {
            "code": "SDK-004",
            "title": "Implement SDK Examples",
            "description": "Create usage examples and tutorials",
            "priority": "low"
        }
    ],
    "testing": [
        {
            "code": "TEST-001",
            "title": "Implement Unit Tests",
            "description": "Create comprehensive unit test suite with 90%+ coverage",
            "priority": "high"
        },
        {
            "code": "TEST-002",
            "title": "Implement Integration Tests",
            "description": "Create integration tests for all major workflows",
            "priority": "high"
        },
        {
            "code": "TEST-003",
            "title": "Implement API Contract Tests",
            "description": "Create contract tests for API compliance",
            "priority": "medium"
        },
        {
            "code": "TEST-004",
            "title": "Implement BDD Scenario Tests",
            "description": "Create behavior-driven development test scenarios",
            "priority": "medium"
        },
        {
            "code": "TEST-005",
            "title": "Implement Performance Tests",
            "description": "Create performance and load testing suite",
            "priority": "medium"
        },
        {
            "code": "QA-001",
            "title": "Code Quality Validation",
            "description": "Run comprehensive code quality checks and validation",
            "priority": "medium"
        },
        {
            "code": "QA-002",
            "title": "Security Scanning",
            "description": "Perform security vulnerability scanning and remediation",
            "priority": "high"
        },
        {
            "code": "QA-003",
            "title": "Performance Profiling",
            "description": "Profile performance and optimize bottlenecks",
            "priority": "medium"
        },
        {
            "code": "QA-004",
            "title": "Documentation Review",
            "description": "Review and validate all documentation completeness",
            "priority": "medium"
        },
        {
            "code": "QA-005",
            "title": "Production Readiness Checklist",
            "description": "Complete production deployment readiness validation",
            "priority": "high"
        }
    ]
}


def create_issue(title: str, body: str, labels: List[str]) -> str:
    """Create a GitHub issue using gh CLI."""
    try:
        cmd = ["gh", "issue", "create", "--title", title, "--body", body]
        for label in labels:
            cmd.extend(["--label", label])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error creating issue '{title}': {e}")
        print(f"Error output: {e.stderr}")
        return ""


def create_package_issues(package_key: str, package_info: Dict) -> List[str]:
    """Create all issues for a specific package."""
    created_issues = []
    
    for category, issues in ISSUE_TEMPLATES.items():
        for issue_template in issues:
            title = f"{issue_template['code']}: {issue_template['title']} for {package_info['name']}"
            
            body = f"""## 📋 Issue Description
{issue_template['description']} for the {package_info['name']} package.

## 🎯 Package Context
**Package**: `{package_key}`
**Purpose**: {package_info['description']}

## 🎯 Acceptance Criteria
- [ ] Implementation follows clean architecture principles
- [ ] Code includes comprehensive error handling
- [ ] Documentation is complete and accurate
- [ ] Unit tests achieve 90%+ coverage
- [ ] Integration with package ecosystem verified
- [ ] Security requirements validated
- [ ] Performance benchmarks met

## 📖 Technical Requirements
- Follow established coding standards and patterns
- Implement proper logging and monitoring
- Include comprehensive input validation
- Ensure thread safety where applicable
- Implement graceful error handling and recovery
- Follow security best practices

## 📁 Expected Deliverables
- Production-ready implementation
- Comprehensive test suite
- API documentation (if applicable)
- User documentation
- Performance benchmarks

## 🔗 Dependencies
Part of {package_info['name']} Epic #{package_info['epic_issue']}

## 📅 Timeline
**Estimated Effort**: 2-5 days (varies by complexity)
**Category**: {category.title()}

## 🏷️ Related Documentation
- Package BRD and requirements documents
- Architecture decision records
- API specifications
- Integration guidelines
"""

            labels = [
                f"package:{package_key}",
                f"category:{category}",
                "type:feature",
                f"priority:{issue_template['priority']}"
            ]
            
            issue_url = create_issue(title, body, labels)
            if issue_url:
                created_issues.append(issue_url)
                print(f"Created: {title}")
            
    return created_issues


def main():
    """Main execution function."""
    print("Creating GitHub issues for 10-package domain expansion...")
    
    all_created_issues = []
    
    for package_key, package_info in PACKAGES.items():
        print(f"\n🚀 Creating issues for {package_info['name']} package...")
        
        package_issues = create_package_issues(package_key, package_info)
        all_created_issues.extend(package_issues)
        
        print(f"✅ Created {len(package_issues)} issues for {package_info['name']}")
    
    print(f"\n🎉 Successfully created {len(all_created_issues)} total issues!")
    print("\n📊 Summary:")
    for package_key, package_info in PACKAGES.items():
        print(f"  - {package_info['name']}: ~30 implementation issues")
    
    print(f"\n🔗 Epic Issues:")
    for package_key, package_info in PACKAGES.items():
        print(f"  - {package_info['name']}: #{package_info['epic_issue']}")


if __name__ == "__main__":
    main()