# Repository Rules and Policies

## 📋 Issue Management Policy

### TODO.md Usage Rule

**STRICT ENFORCEMENT**: The `TODO.md` file is **EXCLUSIVELY** for GitHub CLI reference documentation.

#### ✅ Permitted Content in TODO.md:
- GitHub CLI installation instructions
- GitHub CLI usage examples
- GitHub CLI command reference
- Links to official GitHub documentation
- Repository policy explanations

#### ❌ Prohibited Content in TODO.md:
- Issue tracking or TODO lists
- Project status updates
- Issue summaries or duplicated GitHub data
- Development task lists
- Implementation progress tracking
- Any form of issue management content

#### 🔒 Enforcement:
- All contributors must use GitHub Issues for project management
- Pull requests adding issue tracking content to TODO.md will be rejected
- Regular audits will ensure compliance with this policy
- Violations will require immediate remediation

### Required Workflow for Issue Management

1. **View Issues**: Use `gh issue list` command
2. **Create Issues**: Use `gh issue create` command
3. **Update Issues**: Use `gh issue edit` command
4. **Close Issues**: Use `gh issue close` command
5. **Search Issues**: Use `gh issue list --search` command

### GitHub CLI Installation Requirements

All contributors must have GitHub CLI installed and authenticated:

```bash
# Install GitHub CLI
gh --version

# Authenticate with GitHub
gh auth status

# Test access
gh issue list
```

### Issue Labeling Standards

Use these priority labels for all issues:
- `P0-Critical`: Urgent, blocking issues
- `P1-High`: High priority, important features
- `P2-Medium`: Medium priority, standard features
- `P3-Low`: Low priority, nice-to-have features

### Monitoring and Compliance

- **Automated Checks**: CI/CD pipeline validates TODO.md content
- **Review Process**: All TODO.md changes require approval
- **Documentation**: Any deviation from this policy must be documented
- **Training**: New contributors must acknowledge this policy

## 🚨 Violation Consequences

- **First Violation**: Warning and required remediation
- **Second Violation**: Temporary restriction from TODO.md edits
- **Repeated Violations**: Escalation to project maintainers

## 📞 Support

For questions about this policy or GitHub CLI usage:
- Review the GitHub CLI documentation in TODO.md
- Contact project maintainers
- Check GitHub's official documentation

---

**Policy Version**: 1.0  
**Effective Date**: July 15, 2025  
**Next Review**: Monthly  
**Enforcement**: Immediate