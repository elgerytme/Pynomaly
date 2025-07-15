#!/usr/bin/env python3
"""
Batch GitHub issue creation script for 10-package domain expansion.
Creates issues in smaller batches to avoid API timeouts.
"""

import subprocess
import time
import sys
from typing import List

# Priority packages to start with
PRIORITY_PACKAGES = ["domain_library", "knowledge_graph", "data_architecture"]

def create_issue_batch(package: str, category: str, start_num: int, count: int) -> List[str]:
    """Create a batch of issues for a specific package and category."""
    created_issues = []
    
    # Issue templates by category
    templates = {
        "requirements": [
            ("REQ-001", "Create Business Requirements Document", "high"),
            ("REQ-002", "Define Domain Entities and Models", "high"),
            ("REQ-003", "Create Use Cases and User Stories", "high"),
            ("REQ-004", "Design API Specifications", "high"),
            ("REQ-005", "Create Architecture Decision Records", "medium"),
            ("REQ-006", "Define Data Models and Schemas", "high"),
            ("REQ-007", "Create BDD Scenarios and Story Maps", "medium"),
            ("REQ-008", "Define Security Requirements", "high"),
            ("REQ-009", "Define Performance Requirements", "medium"),
            ("REQ-010", "Create Integration Requirements", "high")
        ],
        "domain": [
            ("DOM-001", "Implement Core Domain Entities", "high"),
            ("DOM-002", "Create Value Objects", "high"),
            ("DOM-003", "Define Domain Services", "high"),
            ("DOM-004", "Implement Repository Abstractions", "high"),
            ("DOM-005", "Create Domain Events", "medium"),
            ("DOM-006", "Implement Business Rules Validation", "high"),
            ("DOM-007", "Create Domain Exceptions", "medium"),
            ("DOM-008", "Implement Aggregate Roots", "high"),
            ("DOM-009", "Create Domain Specifications", "medium"),
            ("DOM-010", "Implement Domain Factories", "medium")
        ],
        "application": [
            ("APP-001", "Create Application Services", "high"),
            ("APP-002", "Implement Use Case Handlers", "high"),
            ("APP-003", "Create DTOs and Mappers", "high"),
            ("APP-004", "Implement Command Handlers", "medium"),
            ("APP-005", "Create Query Handlers", "medium"),
            ("APP-006", "Implement Event Handlers", "medium"),
            ("APP-007", "Create Application Workflows", "medium"),
            ("APP-008", "Implement Cross-Cutting Concerns", "medium"),
            ("APP-009", "Create Application Validators", "medium"),
            ("APP-010", "Implement Application Security", "high")
        ],
        "infrastructure": [
            ("INF-001", "Implement Persistence Layer", "high"),
            ("INF-002", "Create Database Adapters", "high"),
            ("INF-003", "Implement External Service Adapters", "medium"),
            ("INF-004", "Create Monitoring Infrastructure", "medium"),
            ("INF-005", "Implement Security Infrastructure", "high"),
            ("INF-006", "Create Caching Infrastructure", "medium"),
            ("INF-007", "Implement Messaging Infrastructure", "medium"),
            ("INF-008", "Create Logging Infrastructure", "medium"),
            ("INF-009", "Implement Configuration Management", "medium"),
            ("INF-010", "Create Deployment Infrastructure", "medium")
        ],
        "presentation": [
            ("API-001", "Create FastAPI Application Structure", "high"),
            ("API-002", "Implement REST API Endpoints", "high"),
            ("API-003", "Create API Authentication/Authorization", "high"),
            ("CLI-001", "Create Typer CLI Application", "high"),
            ("CLI-002", "Implement CLI Commands", "high"),
            ("WEB-001", "Create React Web UI Components", "medium"),
            ("SDK-001", "Generate Python SDK", "medium"),
            ("SDK-002", "Generate TypeScript SDK", "medium")
        ],
        "testing": [
            ("TEST-001", "Implement Unit Tests", "high"),
            ("TEST-002", "Implement Integration Tests", "high"),
            ("TEST-003", "Implement API Contract Tests", "medium"),
            ("TEST-004", "Implement BDD Scenario Tests", "medium"),
            ("TEST-005", "Implement Performance Tests", "medium")
        ]
    }
    
    if category not in templates:
        print(f"Unknown category: {category}")
        return created_issues
    
    template_list = templates[category]
    end_idx = min(start_num + count, len(template_list))
    
    for i in range(start_num, end_idx):
        if i >= len(template_list):
            break
            
        code, title, priority = template_list[i]
        issue_title = f"{code}: {title} for {package.title().replace('_', ' ')}"
        
        issue_body = f"""## 📋 Issue Description
{title} for the {package.replace('_', ' ').title()} package.

## 🎯 Package Context
**Package**: `{package}`
**Category**: {category.title()}

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

## 📅 Timeline
**Estimated Effort**: 2-5 days (varies by complexity)
**Category**: {category.title()}
**Priority**: {priority.title()}
"""

        labels = [
            f"package:{package}",
            f"category:{category}",
            "type:feature",
            f"priority:{priority}"
        ]
        
        try:
            cmd = ["gh", "issue", "create", "--title", issue_title, "--body", issue_body]
            for label in labels:
                cmd.extend(["--label", label])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            issue_url = result.stdout.strip()
            created_issues.append(issue_url)
            print(f"✅ Created: {issue_title}")
            
            # Small delay to avoid API rate limits
            time.sleep(0.5)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating issue '{issue_title}': {e}")
            print(f"Error output: {e.stderr}")
    
    return created_issues


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python create_batch_issues.py <package_name> [category] [start_num] [count]")
        print("Available packages:", ", ".join(PRIORITY_PACKAGES))
        print("Available categories: requirements, domain, application, infrastructure, presentation, testing")
        return
    
    package = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "requirements"
    start_num = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    print(f"🚀 Creating {count} issues for {package} - {category} category (starting from {start_num})")
    
    created_issues = create_issue_batch(package, category, start_num, count)
    
    print(f"\n🎉 Successfully created {len(created_issues)} issues!")
    for issue_url in created_issues:
        print(f"  - {issue_url}")


if __name__ == "__main__":
    main()