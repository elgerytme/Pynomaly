#!/usr/bin/env python3
"""
Create GitHub Issues for Test Coverage Gaps
Generates specific GitHub issues based on test coverage analysis results.
"""

from datetime import datetime
from pathlib import Path

import click


class GitHubIssueCreator:
    """Creates GitHub issues for test coverage gaps and improvements."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_dir = self.project_root / "issues"
        self.issues_dir.mkdir(exist_ok=True)

    def create_critical_gap_issues(self) -> list[dict]:
        """Create issues for critical test coverage gaps."""
        issues = []

        # Critical Gap 1: CLI Testing
        cli_issue = {
            "title": "[TEST COVERAGE] Critical gap in CLI testing - Only 9.1% coverage",
            "body": self._generate_cli_gap_issue_body(),
            "labels": ["test-coverage", "critical", "cli", "phase-1"],
            "assignees": [],
            "milestone": "Test Coverage Improvement - Phase 1",
        }
        issues.append(cli_issue)

        # Critical Gap 2: Infrastructure Layer Testing
        infra_issue = {
            "title": "[TEST COVERAGE] Critical gap in Infrastructure layer - Only 21% coverage",
            "body": self._generate_infrastructure_gap_issue_body(),
            "labels": ["test-coverage", "critical", "infrastructure", "phase-1"],
            "assignees": [],
            "milestone": "Test Coverage Improvement - Phase 1",
        }
        issues.append(infra_issue)

        # Critical Gap 3: System Testing Missing
        system_issue = {
            "title": "[TEST COVERAGE] Missing system testing category - 0% coverage",
            "body": self._generate_system_testing_issue_body(),
            "labels": ["test-coverage", "critical", "system-testing", "phase-1"],
            "assignees": [],
            "milestone": "Test Coverage Improvement - Phase 1",
        }
        issues.append(system_issue)

        return issues

    def create_high_priority_issues(self) -> list[dict]:
        """Create issues for high-priority improvements."""
        issues = []

        # High Priority 1: Acceptance Testing Framework
        acceptance_issue = {
            "title": "[TEST IMPROVEMENT] Create acceptance testing framework",
            "body": self._generate_acceptance_testing_issue_body(),
            "labels": ["test-improvement", "high", "acceptance-testing", "phase-2"],
            "assignees": [],
            "milestone": "Test Coverage Improvement - Phase 2",
        }
        issues.append(acceptance_issue)

        # High Priority 2: Presentation Layer Enhancement
        presentation_issue = {
            "title": "[TEST IMPROVEMENT] Enhance presentation layer testing (19% → 50%)",
            "body": self._generate_presentation_layer_issue_body(),
            "labels": ["test-improvement", "high", "presentation", "phase-2"],
            "assignees": [],
            "milestone": "Test Coverage Improvement - Phase 2",
        }
        issues.append(presentation_issue)

        # High Priority 3: Cross-Layer Integration Testing
        integration_issue = {
            "title": "[TEST IMPROVEMENT] Implement cross-layer integration testing",
            "body": self._generate_cross_layer_integration_issue_body(),
            "labels": ["test-improvement", "high", "integration", "phase-2"],
            "assignees": [],
            "milestone": "Test Coverage Improvement - Phase 2",
        }
        issues.append(integration_issue)

        return issues

    def create_improvement_task_issues(self) -> list[dict]:
        """Create specific task issues for implementing improvements."""
        issues = []

        # Task issues for CLI improvement
        cli_tasks = [
            {
                "title": "[TEST IMPROVEMENT] CLI Commands - Create comprehensive command-specific tests",
                "body": self._generate_cli_commands_task_body(),
                "labels": ["test-improvement", "cli", "commands", "phase-1-task"],
                "parent": "CLI Testing Gap",
            },
            {
                "title": "[TEST IMPROVEMENT] CLI Integration - Create workflow and configuration tests",
                "body": self._generate_cli_integration_task_body(),
                "labels": ["test-improvement", "cli", "integration", "phase-1-task"],
                "parent": "CLI Testing Gap",
            },
        ]

        # Task issues for Infrastructure improvement
        infra_tasks = [
            {
                "title": "[TEST IMPROVEMENT] Infrastructure Repositories - Create comprehensive repository tests",
                "body": self._generate_repository_testing_task_body(),
                "labels": [
                    "test-improvement",
                    "infrastructure",
                    "repositories",
                    "phase-1-task",
                ],
                "parent": "Infrastructure Layer Gap",
            },
            {
                "title": "[TEST IMPROVEMENT] Infrastructure External Services - Create integration tests",
                "body": self._generate_external_services_task_body(),
                "labels": [
                    "test-improvement",
                    "infrastructure",
                    "external-services",
                    "phase-1-task",
                ],
                "parent": "Infrastructure Layer Gap",
            },
        ]

        issues.extend(cli_tasks + infra_tasks)
        return issues

    def _generate_cli_gap_issue_body(self) -> str:
        """Generate issue body for CLI testing gap."""
        return f"""## Test Coverage Gap Details

### Coverage Information
- **Area/Layer**: CLI (Command Line Interface)
- **Current Coverage**: 9.1% (4 test files / 44 source files)
- **Target Coverage**: 60%
- **Gap Percentage**: 50.9%
- **Priority**: Critical

### Description
The CLI is a major user interface for Pynomaly with severely inadequate test coverage. Only 4 test files exist for 44 source files, creating significant risk for user-facing functionality.

### Impact Assessment
- **Business Impact**: CLI failures could block user workflows and adoption
- **Technical Risk**: Untested command parsing, validation, and execution logic
- **User Impact**: Poor user experience, unreliable command execution

### Affected Components
- [ ] Command-specific implementations (detect, train, autonomous, dataset, export)
- [ ] CLI argument validation and parsing
- [ ] Configuration file processing
- [ ] Help system and documentation
- [ ] Error handling and user feedback
- [ ] Output formatting and export functionality

### Recommended Actions
- [ ] Create comprehensive command-specific test suites
- [ ] Implement CLI workflow integration tests
- [ ] Add argument validation and error handling tests
- [ ] Create help system and documentation validation tests
- [ ] Add configuration file processing tests
- [ ] Implement output formatting validation tests

### Implementation Plan
- **Estimated Effort**: 3 weeks, 1 developer
- **Dependencies**: CLI framework understanding, test data creation
- **Deliverables**:
  ```
  tests/cli/commands/
  ├── test_detect_command.py
  ├── test_train_command.py
  ├── test_autonomous_command.py
  ├── test_dataset_command.py
  └── test_export_command.py

  tests/cli/integration/
  ├── test_cli_workflows.py
  ├── test_cli_configuration.py
  └── test_cli_error_handling.py
  ```

### Acceptance Criteria
- [ ] CLI coverage > 60%
- [ ] All major commands have comprehensive tests
- [ ] Workflow integration tests pass
- [ ] Help system validation complete
- [ ] Error handling and edge cases covered
- [ ] No regression in existing functionality

---
### Analysis Details
**Report Generated**: {datetime.now().isoformat()}
**Analysis Tool**: Automated Test Coverage Analysis
**Priority**: Phase 1 - Critical Gap Resolution"""

    def _generate_infrastructure_gap_issue_body(self) -> str:
        """Generate issue body for infrastructure testing gap."""
        return f"""## Test Coverage Gap Details

### Coverage Information
- **Area/Layer**: Infrastructure Layer
- **Current Coverage**: 21% (54 test files / 254 source files)
- **Target Coverage**: 60%
- **Gap Percentage**: 39%
- **Priority**: Critical

### Description
The infrastructure layer has significant gaps in test coverage, particularly around database operations, external service integrations, and caching strategies. This creates risks for data persistence and system reliability.

### Impact Assessment
- **Business Impact**: Data persistence failures, external API integration issues
- **Technical Risk**: Database corruption, cache inconsistency, service communication failures
- **User Impact**: System unreliability, data loss, performance degradation

### Affected Components
- [ ] Database repositories and persistence
- [ ] External service integrations (APIs, third-party services)
- [ ] Caching layer and strategies
- [ ] Message queues and streaming
- [ ] Configuration management
- [ ] Monitoring and alerting systems
- [ ] Performance optimization components

### Recommended Actions
- [ ] Create comprehensive repository testing framework
- [ ] Implement external service integration tests with proper mocking
- [ ] Add caching layer validation and invalidation tests
- [ ] Create message queue and streaming integration tests
- [ ] Implement configuration management testing
- [ ] Add monitoring and metrics collection tests

### Implementation Plan
- **Estimated Effort**: 4 weeks, 2 developers
- **Dependencies**: Test database setup, external service mocking, containerization
- **Deliverables**:
  ```
  tests/infrastructure/
  ├── persistence/
  │   ├── test_repositories.py
  │   ├── test_database_integration.py
  │   └── test_data_persistence.py
  ├── cache/
  │   ├── test_caching_strategies.py
  │   └── test_cache_invalidation.py
  ├── external/
  │   ├── test_service_integrations.py
  │   └── test_api_clients.py
  └── monitoring/
      ├── test_metrics_collection.py
      └── test_health_monitoring.py
  ```

### Acceptance Criteria
- [ ] Infrastructure coverage > 60%
- [ ] Database operations comprehensively tested
- [ ] External service integrations validated with proper mocking
- [ ] Caching strategies and invalidation tested
- [ ] Monitoring and metrics collection verified
- [ ] Performance and reliability maintained

---
### Analysis Details
**Report Generated**: {datetime.now().isoformat()}
**Analysis Tool**: Automated Test Coverage Analysis
**Priority**: Phase 1 - Critical Gap Resolution"""

    def _generate_system_testing_issue_body(self) -> str:
        """Generate issue body for system testing gap."""
        return f"""## Test Coverage Gap Details

### Coverage Information
- **Area/Layer**: System Testing
- **Current Coverage**: 0% (No dedicated system test directory)
- **Target Coverage**: Complete system test suite
- **Gap Percentage**: 100%
- **Priority**: Critical

### Description
There is no dedicated system testing category, which means end-to-end system validation is missing. This creates significant risk for system integration failures and deployment issues.

### Impact Assessment
- **Business Impact**: System integration failures, deployment issues, poor system reliability
- **Technical Risk**: Component integration failures, system-level bugs, deployment problems
- **User Impact**: Complete system failures, unreliable user experience

### Affected Components
- [ ] End-to-end anomaly detection workflows
- [ ] Multi-component integration points
- [ ] System configuration and deployment
- [ ] Cross-service communication
- [ ] System performance under load
- [ ] Recovery and failover procedures

### Recommended Actions
- [ ] Create system testing framework and directory structure
- [ ] Implement end-to-end anomaly detection workflow tests
- [ ] Add system integration point validation
- [ ] Create deployment and configuration validation tests
- [ ] Implement system performance and load tests
- [ ] Add system recovery and failover tests

### Implementation Plan
- **Estimated Effort**: 2 weeks, 1 developer
- **Dependencies**: System understanding, test environment setup, deployment automation
- **Deliverables**:
  ```
  tests/system/
  ├── test_e2e_anomaly_detection.py
  ├── test_system_integration.py
  ├── test_deployment_validation.py
  ├── test_system_performance.py
  ├── test_system_recovery.py
  └── conftest.py
  ```

### Acceptance Criteria
- [ ] System test directory and framework created
- [ ] End-to-end workflows comprehensively tested
- [ ] System integration points validated
- [ ] Deployment scenarios covered
- [ ] Performance and load testing implemented
- [ ] Recovery and failover procedures tested

---
### Analysis Details
**Report Generated**: {datetime.now().isoformat()}
**Analysis Tool**: Automated Test Coverage Analysis
**Priority**: Phase 1 - Critical Gap Resolution"""

    def _generate_acceptance_testing_issue_body(self) -> str:
        """Generate issue body for acceptance testing."""
        return """## Test Improvement Task

### Component Information
- **Component**: Acceptance Testing Framework
- **Layer**: Cross-Layer
- **Area**: Quality Assurance
- **Current State**: No formal acceptance testing framework

### Task Description
Create a comprehensive acceptance testing framework to validate user stories and business requirements from a user perspective. This will ensure that features meet stakeholder expectations and business goals.

### Scope
**Included:**
- [ ] User story validation framework
- [ ] Business requirement compliance testing
- [ ] Feature acceptance criteria verification
- [ ] Stakeholder scenario testing
- [ ] BDD integration for acceptance tests

**Excluded:**
- [ ] Unit or integration test replacement
- [ ] Performance testing (separate concern)
- [ ] Security testing (separate framework)

### Implementation Details

#### Test Files to Create/Modify
```
tests/acceptance/
├── user_stories/
│   ├── test_anomaly_detection_user_stories.py
│   ├── test_data_management_user_stories.py
│   └── test_reporting_user_stories.py
├── business_requirements/
│   ├── test_compliance_requirements.py
│   └── test_functional_requirements.py
├── feature_acceptance/
│   ├── test_feature_completeness.py
│   └── test_feature_quality.py
└── stakeholder_scenarios/
    ├── test_data_scientist_workflows.py
    └── test_business_analyst_workflows.py
```

#### Testing Approach
- **Test Type**: Acceptance/BDD
- **Framework**: pytest-bdd, Gherkin scenarios
- **Mocking Strategy**: Minimal mocking, focus on real user scenarios
- **Data Strategy**: Realistic test data representing actual use cases

#### Specific Test Scenarios
- [ ] User story completion validation
- [ ] Business requirement compliance
- [ ] Feature acceptance criteria verification
- [ ] End-user workflow validation
- [ ] Stakeholder scenario testing
- [ ] User experience validation

### Expected Outcomes
- **Coverage Improvement**: Complete user story and requirement coverage
- **Quality Metrics**: All features validated against acceptance criteria
- **Business Value**: Ensured alignment with business goals

### Dependencies
- [ ] User story documentation
- [ ] Business requirements specification
- [ ] Stakeholder input and validation
- [ ] BDD framework setup

### Definition of Done
- [ ] Acceptance testing framework operational
- [ ] All current user stories have acceptance tests
- [ ] Business requirements are validated
- [ ] Stakeholder scenarios are covered
- [ ] CI/CD integration complete
- [ ] Documentation and guidelines created

### Effort Estimation
- **Complexity**: Medium
- **Estimated Time**: 2 weeks
- **Skills Required**: BDD experience, business analysis, user story understanding

---
### Tracking
- **Parent Epic**: Test Coverage Improvement - Phase 2
- **Sprint**: Phase 2 Implementation
- **Priority**: High""".format()

    def _generate_presentation_layer_issue_body(self) -> str:
        """Generate issue body for presentation layer improvement."""
        return """## Test Improvement Task

### Component Information
- **Component**: Presentation Layer (API, CLI, Web UI)
- **Layer**: Presentation
- **Area**: User Interfaces
- **Current State**: 19% coverage (22 test files / 115 source files)

### Task Description
Enhance presentation layer testing to improve coverage from 19% to 50%. Focus on API endpoints, CLI interfaces, and web UI components that directly interact with users.

### Scope
**Included:**
- [ ] Web interface component testing
- [ ] API endpoint comprehensive testing
- [ ] Authentication and authorization testing
- [ ] Input validation and sanitization
- [ ] Error page and handling testing
- [ ] Response formatting validation

**Excluded:**
- [ ] UI automation (already well covered)
- [ ] CLI command testing (separate epic)
- [ ] Performance testing (separate concern)

### Implementation Details

#### Test Files to Create/Modify
```
tests/presentation/
├── api/
│   ├── test_authentication_endpoints.py
│   ├── test_data_endpoints.py
│   ├── test_model_endpoints.py
│   └── test_error_handling.py
├── web/
│   ├── test_web_components.py
│   ├── test_template_rendering.py
│   └── test_user_interactions.py
└── shared/
    ├── test_input_validation.py
    └── test_response_formatting.py
```

#### Testing Approach
- **Test Type**: Unit and Integration
- **Framework**: pytest, FastAPI TestClient
- **Mocking Strategy**: Mock external dependencies, test interface contracts
- **Data Strategy**: Comprehensive request/response validation

#### Specific Test Scenarios
- [ ] API endpoint request/response validation
- [ ] Authentication and authorization edge cases
- [ ] Input validation and error handling
- [ ] Response formatting and serialization
- [ ] Error page rendering and messaging
- [ ] User interface component behavior

### Expected Outcomes
- **Coverage Improvement**: 19% → 50% presentation layer coverage
- **Quality Metrics**: All user-facing interfaces validated
- **Security**: Input validation and security testing complete

### Dependencies
- [ ] API specification documentation
- [ ] Authentication system understanding
- [ ] Web framework knowledge
- [ ] Security testing tools

### Definition of Done
- [ ] Presentation layer coverage > 50%
- [ ] All critical API endpoints tested
- [ ] Authentication/authorization validated
- [ ] Input validation comprehensive
- [ ] Error handling verified
- [ ] No regression in existing tests

### Effort Estimation
- **Complexity**: Medium
- **Estimated Time**: 2 weeks
- **Skills Required**: FastAPI testing, web development, security testing

---
### Tracking
- **Parent Epic**: Test Coverage Improvement - Phase 2
- **Sprint**: Phase 2 Implementation
- **Priority**: High""".format()

    def _generate_cross_layer_integration_issue_body(self) -> str:
        """Generate issue body for cross-layer integration testing."""
        return """## Test Improvement Task

### Component Information
- **Component**: Cross-Layer Integration
- **Layer**: Integration
- **Area**: System Architecture
- **Current State**: Limited cross-layer boundary testing

### Task Description
Implement comprehensive cross-layer integration testing to validate boundaries and communication between architectural layers. Ensure proper separation of concerns and contract compliance.

### Scope
**Included:**
- [ ] Domain ↔ Application boundary testing
- [ ] Application ↔ Infrastructure integration
- [ ] Infrastructure ↔ Presentation integration
- [ ] Cross-service communication testing
- [ ] Contract validation between layers
- [ ] Error propagation testing

**Excluded:**
- [ ] Unit testing within layers
- [ ] End-to-end system testing
- [ ] Performance testing

### Implementation Details

#### Test Files to Create/Modify
```
tests/integration/cross_layer/
├── test_domain_application_integration.py
├── test_application_infrastructure_integration.py
├── test_infrastructure_presentation_integration.py
├── test_service_communication.py
├── test_contract_compliance.py
└── test_error_propagation.py
```

#### Testing Approach
- **Test Type**: Integration
- **Framework**: pytest with dependency injection
- **Mocking Strategy**: Mock external boundaries, test internal integration
- **Data Strategy**: Realistic data flow testing

#### Specific Test Scenarios
- [ ] Layer boundary contract validation
- [ ] Data transformation between layers
- [ ] Error handling and propagation
- [ ] Service communication protocols
- [ ] Dependency injection validation
- [ ] Interface compliance testing

### Expected Outcomes
- **Coverage Improvement**: Comprehensive cross-layer integration coverage
- **Quality Metrics**: All layer boundaries validated
- **Architecture**: Clean architecture compliance verified

### Dependencies
- [ ] Architecture documentation
- [ ] Interface specifications
- [ ] Dependency injection framework
- [ ] Contract testing tools

### Definition of Done
- [ ] All layer boundaries tested
- [ ] Contract compliance validated
- [ ] Error propagation verified
- [ ] Service communication tested
- [ ] Integration points documented
- [ ] No architectural violations detected

### Effort Estimation
- **Complexity**: High
- **Estimated Time**: 2 weeks
- **Skills Required**: Clean architecture, integration testing, system design

---
### Tracking
- **Parent Epic**: Test Coverage Improvement - Phase 2
- **Sprint**: Phase 2 Implementation
- **Priority**: High""".format()

    def _generate_cli_commands_task_body(self) -> str:
        """Generate task body for CLI commands testing."""
        return """## Test Improvement Task

### Component Information
- **Component**: CLI Commands
- **Layer**: Presentation
- **Area**: CLI
- **Current State**: Basic CLI structure testing only

### Task Description
Create comprehensive test suites for all CLI commands including detect, train, autonomous, dataset, and export commands. Ensure proper argument validation, command execution, and error handling.

### Scope
**Included:**
- [ ] All major CLI commands (detect, train, autonomous, dataset, export)
- [ ] Argument parsing and validation
- [ ] Command execution logic
- [ ] Error handling and user feedback
- [ ] Help system validation
- [ ] Output formatting

**Excluded:**
- [ ] CLI framework testing (typer framework)
- [ ] Integration workflows (separate task)
- [ ] Configuration file processing (separate task)

### Implementation Details

#### Test Files to Create/Modify
```
tests/cli/commands/
├── test_detect_command.py
├── test_train_command.py
├── test_autonomous_command.py
├── test_dataset_command.py
├── test_export_command.py
└── test_help_system.py
```

#### Testing Approach
- **Test Type**: Unit
- **Framework**: pytest with CLI testing utilities
- **Mocking Strategy**: Mock core services, test CLI interface
- **Data Strategy**: Command-line argument simulation

#### Specific Test Scenarios
- [ ] Valid command execution with correct arguments
- [ ] Invalid argument handling and error messages
- [ ] Help text generation and accuracy
- [ ] Output formatting and structure
- [ ] Edge cases and boundary conditions
- [ ] Command composition and chaining

### Expected Outcomes
- **Coverage Improvement**: Major improvement in CLI command coverage
- **Quality Metrics**: All commands properly validated
- **User Experience**: Reliable command execution

### Dependencies
- [ ] CLI framework understanding (typer)
- [ ] Command argument specifications
- [ ] Output format requirements
- [ ] Error message standards

### Definition of Done
- [ ] All major commands have comprehensive tests
- [ ] Argument validation tested
- [ ] Error handling verified
- [ ] Help system validated
- [ ] Output formatting checked
- [ ] No regression in CLI functionality

### Effort Estimation
- **Complexity**: Medium
- **Estimated Time**: 1.5 weeks
- **Skills Required**: CLI testing, typer framework, argument parsing

---
### Tracking
- **Parent Epic**: CLI Testing Gap
- **Sprint**: Phase 1 Implementation
- **Priority**: Critical""".format()

    def _generate_cli_integration_task_body(self) -> str:
        """Generate task body for CLI integration testing."""
        return """## Test Improvement Task

### Component Information
- **Component**: CLI Integration
- **Layer**: Presentation
- **Area**: CLI
- **Current State**: No CLI workflow integration testing

### Task Description
Create comprehensive CLI workflow integration tests that validate end-to-end CLI operations, configuration processing, and multi-command workflows.

### Scope
**Included:**
- [ ] Multi-command workflow testing
- [ ] Configuration file processing
- [ ] CLI pipeline operations
- [ ] Error recovery in workflows
- [ ] Output chaining between commands
- [ ] Environment variable handling

**Excluded:**
- [ ] Individual command testing (separate task)
- [ ] System-level integration (separate concern)
- [ ] Performance testing

### Implementation Details

#### Test Files to Create/Modify
```
tests/cli/integration/
├── test_cli_workflows.py
├── test_cli_configuration.py
├── test_cli_error_handling.py
├── test_cli_output_chaining.py
└── test_cli_environment.py
```

#### Testing Approach
- **Test Type**: Integration
- **Framework**: pytest with subprocess testing
- **Mocking Strategy**: Minimal mocking, test real CLI execution
- **Data Strategy**: Temporary files and directories for testing

#### Specific Test Scenarios
- [ ] Complete anomaly detection workflow via CLI
- [ ] Configuration file loading and validation
- [ ] Error recovery and graceful degradation
- [ ] Output file generation and validation
- [ ] Environment variable configuration
- [ ] Multi-step data processing pipelines

### Expected Outcomes
- **Coverage Improvement**: Complete CLI workflow validation
- **Quality Metrics**: Reliable end-to-end CLI operations
- **User Experience**: Smooth workflow execution

### Dependencies
- [ ] CLI command testing completion
- [ ] Configuration file specifications
- [ ] Workflow documentation
- [ ] Test data creation

### Definition of Done
- [ ] All major workflows tested
- [ ] Configuration processing validated
- [ ] Error handling verified
- [ ] Output chaining working
- [ ] Environment handling tested
- [ ] No regression in CLI workflows

### Effort Estimation
- **Complexity**: Medium
- **Estimated Time**: 1 week
- **Skills Required**: Integration testing, CLI workflows, subprocess testing

---
### Tracking
- **Parent Epic**: CLI Testing Gap
- **Sprint**: Phase 1 Implementation
- **Priority**: Critical""".format()

    def _generate_repository_testing_task_body(self) -> str:
        """Generate task body for repository testing."""
        return """## Test Improvement Task

### Component Information
- **Component**: Infrastructure Repositories
- **Layer**: Infrastructure
- **Area**: Data Persistence
- **Current State**: Limited repository testing

### Task Description
Create comprehensive repository testing covering database operations, data persistence patterns, and repository interface compliance.

### Scope
**Included:**
- [ ] Repository interface implementations
- [ ] Database CRUD operations
- [ ] Query optimization and performance
- [ ] Transaction handling
- [ ] Data consistency validation
- [ ] Error handling and recovery

**Excluded:**
- [ ] Database schema migrations
- [ ] Performance tuning (separate concern)
- [ ] External service integration

### Implementation Details

#### Test Files to Create/Modify
```
tests/infrastructure/persistence/
├── test_repositories.py
├── test_database_integration.py
├── test_data_persistence.py
├── test_transaction_handling.py
└── test_query_optimization.py
```

#### Testing Approach
- **Test Type**: Integration
- **Framework**: pytest with database fixtures
- **Mocking Strategy**: Test database with real operations
- **Data Strategy**: Isolated test database with controlled data

#### Specific Test Scenarios
- [ ] Repository CRUD operations
- [ ] Complex query execution
- [ ] Transaction rollback and commit
- [ ] Data consistency validation
- [ ] Error handling for database failures
- [ ] Concurrent access testing

### Expected Outcomes
- **Coverage Improvement**: Complete repository operation validation
- **Quality Metrics**: Data integrity and reliability assured
- **Performance**: Query optimization validated

### Dependencies
- [ ] Test database setup
- [ ] Repository interface specifications
- [ ] Data model understanding
- [ ] Transaction management knowledge

### Definition of Done
- [ ] All repository operations tested
- [ ] Database integration validated
- [ ] Transaction handling verified
- [ ] Data consistency assured
- [ ] Error scenarios covered
- [ ] Performance benchmarks met

### Effort Estimation
- **Complexity**: High
- **Estimated Time**: 2 weeks
- **Skills Required**: Database testing, repository patterns, transaction management

---
### Tracking
- **Parent Epic**: Infrastructure Layer Gap
- **Sprint**: Phase 1 Implementation
- **Priority**: Critical""".format()

    def _generate_external_services_task_body(self) -> str:
        """Generate task body for external services testing."""
        return """## Test Improvement Task

### Component Information
- **Component**: Infrastructure External Services
- **Layer**: Infrastructure
- **Area**: External Integrations
- **Current State**: Limited external service integration testing

### Task Description
Create comprehensive external service integration tests with proper mocking, contract validation, and error handling for all third-party service interactions.

### Scope
**Included:**
- [ ] External API client testing
- [ ] Service contract validation
- [ ] Error handling and retry logic
- [ ] Authentication and authorization
- [ ] Rate limiting and throttling
- [ ] Circuit breaker patterns

**Excluded:**
- [ ] Actual third-party service testing
- [ ] Performance load testing
- [ ] Service configuration management

### Implementation Details

#### Test Files to Create/Modify
```
tests/infrastructure/external/
├── test_service_integrations.py
├── test_api_clients.py
├── test_authentication_handlers.py
├── test_error_handling.py
└── test_circuit_breakers.py
```

#### Testing Approach
- **Test Type**: Integration with mocking
- **Framework**: pytest with requests-mock, responses
- **Mocking Strategy**: Mock external services, test client behavior
- **Data Strategy**: Simulated service responses and error conditions

#### Specific Test Scenarios
- [ ] Successful API calls and response handling
- [ ] Authentication token management
- [ ] Error response handling and recovery
- [ ] Rate limiting and retry logic
- [ ] Circuit breaker activation and recovery
- [ ] Timeout handling and fallbacks

### Expected Outcomes
- **Coverage Improvement**: Complete external service interaction validation
- **Quality Metrics**: Reliable external service integration
- **Resilience**: Proper error handling and recovery

### Dependencies
- [ ] External service specifications
- [ ] Authentication requirements
- [ ] Error handling patterns
- [ ] Mocking framework setup

### Definition of Done
- [ ] All external service clients tested
- [ ] Authentication mechanisms validated
- [ ] Error handling comprehensive
- [ ] Retry logic verified
- [ ] Circuit breakers tested
- [ ] Fallback mechanisms working

### Effort Estimation
- **Complexity**: High
- **Estimated Time**: 2 weeks
- **Skills Required**: API testing, mocking, resilience patterns

---
### Tracking
- **Parent Epic**: Infrastructure Layer Gap
- **Sprint**: Phase 1 Implementation
- **Priority**: Critical""".format()

    def save_issues_to_files(self, issues: list[dict], category: str) -> None:
        """Save issues to markdown files for review."""
        category_dir = self.issues_dir / category
        category_dir.mkdir(exist_ok=True)

        for i, issue in enumerate(issues, 1):
            filename = f"{i:02d}_{issue['title'].replace('[', '').replace(']', '').replace(' ', '_').replace('/', '_')[:50]}.md"
            filepath = category_dir / filename

            content = f"""# {issue['title']}

**Labels**: {', '.join(issue['labels'])}
**Assignees**: {', '.join(issue.get('assignees', []))}
**Milestone**: {issue.get('milestone', 'N/A')}

{issue['body']}
"""
            with open(filepath, "w") as f:
                f.write(content)

        click.echo(f"✅ Saved {len(issues)} {category} issues to {category_dir}")

    def generate_all_issues(self) -> None:
        """Generate all GitHub issues."""
        click.echo("📋 Creating GitHub issues for test coverage improvements...")

        # Generate critical gap issues
        critical_issues = self.create_critical_gap_issues()
        self.save_issues_to_files(critical_issues, "critical_gaps")

        # Generate high priority issues
        high_priority_issues = self.create_high_priority_issues()
        self.save_issues_to_files(high_priority_issues, "high_priority")

        # Generate task issues
        task_issues = self.create_improvement_task_issues()
        self.save_issues_to_files(task_issues, "improvement_tasks")

        # Generate summary
        total_issues = (
            len(critical_issues) + len(high_priority_issues) + len(task_issues)
        )

        summary = f"""# GitHub Issues Summary

**Generated**: {datetime.now().isoformat()}
**Total Issues**: {total_issues}

## Categories
- **Critical Gaps**: {len(critical_issues)} issues
- **High Priority**: {len(high_priority_issues)} issues
- **Improvement Tasks**: {len(task_issues)} issues

## Next Steps
1. Review all generated issues in the `issues/` directory
2. Create these issues in GitHub manually or using GitHub CLI
3. Assign issues to team members
4. Set up milestones for Phase 1 and Phase 2
5. Begin implementation according to the improvement plan

## GitHub CLI Commands
To create these issues using GitHub CLI:

```bash
# Create critical gap issues
{self._generate_gh_cli_commands(critical_issues)}

# Create high priority issues
{self._generate_gh_cli_commands(high_priority_issues)}

# Create task issues
{self._generate_gh_cli_commands(task_issues)}
```
"""

        with open(self.issues_dir / "README.md", "w") as f:
            f.write(summary)

        click.echo("\n📊 Issue Generation Summary:")
        click.echo(f"├── Critical Gaps: {len(critical_issues)} issues")
        click.echo(f"├── High Priority: {len(high_priority_issues)} issues")
        click.echo(f"├── Improvement Tasks: {len(task_issues)} issues")
        click.echo(f"└── Total: {total_issues} issues")
        click.echo(f"\n✅ All issues saved to: {self.issues_dir}")

    def _generate_gh_cli_commands(self, issues: list[dict]) -> str:
        """Generate GitHub CLI commands for creating issues."""
        commands = []
        for issue in issues:
            labels = " ".join([f'--label "{label}"' for label in issue["labels"]])
            title = issue["title"].replace('"', '\\"')

            # Save body to temp file and reference it
            commands.append(
                f'gh issue create --title "{title}" {labels} --body-file <(echo "{issue["body"][:100]}...")'
            )

        return "\n".join(commands[:3])  # Show first 3 as examples


@click.command()
@click.option("--project-root", default=".", help="Project root directory")
def main(project_root: str):
    """Create GitHub issues for test coverage improvements."""
    creator = GitHubIssueCreator(project_root)
    creator.generate_all_issues()


if __name__ == "__main__":
    main()
