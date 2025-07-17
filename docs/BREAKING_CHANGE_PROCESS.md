# Breaking Change Exception Process

## Overview
This document defines the process for handling breaking changes that require coordinated updates across multiple packages, which is an exception to the Single Package Development Rule.

## When Breaking Changes Are Justified

### Legitimate Scenarios
1. **API Contract Changes**: Interface definitions that affect multiple packages
2. **Shared Schema Updates**: Database or data structure changes
3. **Security Updates**: Critical security patches requiring coordinated updates
4. **Major Version Releases**: Planned breaking changes across the ecosystem
5. **Dependency Updates**: Major version updates of shared dependencies

### Examples of Valid Breaking Changes
- Changing a shared interface signature used across packages
- Updating database schema that affects multiple data access layers
- Upgrading a core dependency that requires API changes
- Refactoring shared data structures or protocols

## Exception Process

### Phase 1: Planning and Approval
1. **RFC Creation**: Create a Request for Comments document
2. **Impact Assessment**: Analyze all affected packages
3. **Architecture Review**: Get approval from architecture team
4. **Timeline Planning**: Establish coordinated development timeline

### Phase 2: Documentation
1. **Create Justification**: Write `BREAKING_CHANGE_JUSTIFICATION.md`
2. **Document Changes**: Detail all required modifications
3. **Rollback Plan**: Define clear rollback strategy
4. **Testing Strategy**: Plan comprehensive testing approach

### Phase 3: Implementation
1. **Coordinated Development**: Make minimal required changes
2. **Atomic Commits**: All changes in single commit or PR
3. **Comprehensive Testing**: Validate all affected packages
4. **Documentation Updates**: Update all relevant documentation

## Required Documentation

### BREAKING_CHANGE_JUSTIFICATION.md Template
```markdown
# Breaking Change Justification

## Change Summary
Brief description of the breaking change and why it's necessary.

## Affected Packages
- src/packages/package1/ - Description of changes
- src/packages/package2/ - Description of changes

## Impact Assessment
- **API Changes**: List of modified interfaces
- **Data Changes**: Database or schema modifications
- **Behavioral Changes**: Modified functionality

## Justification
Detailed explanation of why this coordinated update is necessary.

## Rollback Plan
Step-by-step process to revert changes if needed.

## Testing Strategy
- Unit tests for each affected package
- Integration tests for cross-package interactions
- Performance impact assessment

## Timeline
- Development: [dates]
- Testing: [dates]
- Deployment: [dates]

## Approval
- Architecture Team: [signature/date]
- Security Team: [signature/date] (if applicable)
```

## Validation Checklist

### Pre-Implementation
- [ ] RFC approved by architecture team
- [ ] Impact assessment completed
- [ ] BREAKING_CHANGE_JUSTIFICATION.md created
- [ ] Rollback plan documented
- [ ] Testing strategy defined

### Implementation
- [ ] Changes are minimal and focused
- [ ] All affected packages compile
- [ ] Unit tests pass for all packages
- [ ] Integration tests pass
- [ ] Documentation updated

### Post-Implementation
- [ ] All validation tools pass
- [ ] Performance impact assessed
- [ ] Monitoring in place
- [ ] Team notifications sent

## Commit Message Format
Breaking changes must use conventional commit format:
```
feat!: update shared detection interface

BREAKING CHANGE: Modified DetectionInterface to support new result format

- Updated src/packages/data/anomaly_detection/
- Updated src/packages/software/interfaces/
- Updated src/packages/software/core/

Closes #123
```

## Monitoring and Metrics

### Success Metrics
- Exception frequency (target: <5% of all commits)
- Rollback rate (target: <10% of breaking changes)
- Time to resolution (target: <24 hours)
- Package stability post-change

### Failure Indicators
- Frequent breaking changes
- High rollback rates
- Extended downtime
- Package instability

## Governance

### Architecture Review Board
- Reviews all breaking change RFCs
- Approves coordinated updates
- Monitors exception frequency
- Updates process based on lessons learned

### Exception Tracking
- All breaking changes logged
- Regular review of exception patterns
- Process improvement recommendations
- Quarterly governance reports

## Tools and Automation

### Validation Tools
- `validate_single_package_development.py` - Detects breaking changes
- `validate_breaking_change_justification.py` - Validates documentation
- Pre-commit hooks for automatic validation

### CI/CD Integration
Breaking changes trigger additional validation:
- Extended test suite across all packages
- Security scanning
- Performance impact analysis
- Documentation validation

## Emergency Procedures

### Critical Security Updates
For urgent security patches:
1. Security team approval required
2. Abbreviated documentation acceptable
3. Post-hoc justification within 24 hours
4. Full retroactive compliance review

### Hotfix Process
For production-critical issues:
1. Incident commander approval
2. Minimal viable fix approach
3. Follow-up comprehensive solution
4. Post-incident review and documentation

---

This process ensures that breaking changes are rare, well-justified, and properly managed while maintaining the benefits of single-package development.