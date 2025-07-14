# Changelog Automation Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md) > 📄 Changelog Automation Guide

---


This guide explains the comprehensive changelog automation system implemented for Pynomaly, ensuring that all logical units of work are properly documented in the project's version history.

## Overview

The changelog automation system ensures that:
- **CHANGELOG.md** is updated whenever significant changes are made
- **Version history** is properly maintained with semantic versioning
- **Release notes** are comprehensive and user-friendly
- **Development workflow** includes proper documentation practices

## System Components

### 1. Rules and Guidelines (CLAUDE.md)

The foundation of the system is the **Changelog Management Rules** section in `CLAUDE.md`, which defines:

- **What constitutes a logical unit of work**
- **When changelog updates are mandatory**
- **Entry format and categories**
- **Integration with development workflow**

### 2. Validation Scripts

#### `scripts/check_changelog_update.py`
**Purpose**: Validates that CHANGELOG.md has been updated for significant changes.

**Features**:
- Detects significant changes based on file paths and line count
- Analyzes git diff to determine if changelog update is required
- Provides detailed feedback and suggestions
- Can be used as pre-commit hook or CI check

**Usage**:
```bash
python3 scripts/check_changelog_update.py
```

#### `scripts/check_changelog_update_pr.py`
**Purpose**: Special version for GitHub Actions PR validation.

**Features**:
- Analyzes PR changes from file lists
- Categorizes changes by type
- Provides specific suggestions for changelog sections

### 3. Interactive Helper Tools

#### `scripts/update_changelog_helper.py`
**Purpose**: Interactive tool for creating properly formatted changelog entries.

**Features**:
- Guides users through changelog creation process
- Handles semantic versioning automatically
- Enforces proper formatting and categories
- Validates changelog entries before adding

**Usage**:
```bash
python3 scripts/update_changelog_helper.py
```

**Workflow**:
1. Detects current version from CHANGELOG.md
2. Prompts for change type (major/minor/patch)
3. Collects changelog entries by category
4. Formats entry according to project standards
5. Updates CHANGELOG.md with new entry

### 4. Git Integration

#### Pre-commit Hook (`scripts/pre_commit_changelog_hook.sh`)
**Purpose**: Automatically check changelog updates before commits.

**Features**:
- Runs changelog checker on every commit
- Blocks commits that require changelog updates
- Provides helpful feedback and suggestions
- Can be bypassed with `--no-verify` if needed

#### Git Aliases
**Purpose**: Convenient commands for changelog operations.

**Available aliases**:
```bash
git changelog-update     # Interactive changelog helper
git changelog-check      # Check if update needed  
git changelog-recent     # View recent entries
git commit-bypass        # Commit without hooks
```

### 5. CI/CD Integration

#### GitHub Actions (`.github/workflows/changelog-check.yml`)
**Purpose**: Automated PR validation for changelog updates.

**Features**:
- Checks all PRs for significant changes
- Validates changelog format when updated
- Posts helpful comments on PRs
- Prevents merging without proper changelog updates

#### Pull Request Template (`.github/pull_request_template.md`)
**Purpose**: Enforces changelog requirements in PR process.

**Features**:
- Mandatory changelog update checklist
- Change significance assessment
- Changelog entry preview section
- Integration with CI validation

### 6. Installation and Setup

#### `scripts/install_changelog_hooks.sh`
**Purpose**: One-command setup for all changelog automation.

**Installs**:
- Pre-commit hooks
- Git aliases
- Convenience scripts
- Shell completions (optional)

**Usage**:
```bash
./scripts/install_changelog_hooks.sh
```

## What Triggers Changelog Updates

### Mandatory Updates
Changelog updates are **required** for:

- ✅ **Feature Implementation**: Complete new features with tests and documentation
- ✅ **Bug Fixes**: Resolved issues that affect functionality  
- ✅ **Infrastructure Changes**: CI/CD, Docker, deployment configuration updates
- ✅ **Documentation Phases**: Major documentation additions or restructuring
- ✅ **Testing Milestones**: Significant test coverage improvements
- ✅ **Algorithm Implementations**: New ML algorithms, adapters, or detection methods
- ✅ **Performance Improvements**: Optimization work with measurable improvements
- ✅ **Security Enhancements**: Authentication, authorization, or security fixes
- ✅ **API Changes**: Breaking or non-breaking API modifications
- ✅ **Dependency Updates**: Major dependency upgrades or additions

### Change Detection Logic
The system considers changes **significant** if:

1. **Critical files modified**: `src/`, `examples/`, `docs/`, `scripts/`, `tests/`, `docker/`
2. **Configuration changes**: `pyproject.toml`, `requirements.txt`, `Dockerfile`
3. **Documentation updates**: `README.md`, documentation files
4. **Line count threshold**: More than 20 lines changed in critical paths

### Excluded Files
These files **don't require** changelog updates:
- `.gitignore`, `.github/` (except workflows)
- `TODO.md`, `CLAUDE.md` (project management files)
- Cache directories, build artifacts
- `CHANGELOG.md` itself

## Changelog Entry Format

### Standard Template
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features, capabilities, or functionality
- Algorithm implementations with performance characteristics
- Documentation sections with scope and audience

### Changed
- Modified functionality with migration guidance
- Updated dependencies with version information
- Improved performance with benchmark results

### Fixed
- Bug fixes with issue references and impact
- Security vulnerabilities with severity assessment
- Compatibility issues with affected systems

### Documentation
- New guides, tutorials, or reference materials
- Updated existing documentation with scope
- API documentation improvements

### Infrastructure
- CI/CD pipeline improvements
- Docker configuration enhancements
- Deployment automation additions

### Testing
- Test coverage improvements with percentages
- New test infrastructure or frameworks
- Performance test additions
```

### Categories Explained

- **Added**: New features, capabilities, or functionality
- **Changed**: Changes in existing functionality or behavior
- **Deprecated**: Soon-to-be removed features (with timeline)
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes and issue resolutions
- **Security**: Security-related changes and vulnerability fixes
- **Performance**: Performance improvements and optimizations
- **Documentation**: Documentation additions, improvements, or restructuring
- **Infrastructure**: CI/CD, build system, or deployment changes
- **Testing**: Test additions, improvements, or infrastructure changes

## Semantic Versioning

The system follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes, major features
- **MINOR** (0.X.0): New features, backwards compatible
- **PATCH** (0.0.X): Bug fixes, small improvements

### Version Increment Guidelines

| Change Type | Version Impact | Examples |
|-------------|----------------|----------|
| Breaking API changes | MAJOR | Remove functions, change signatures |
| New features | MINOR | Add algorithms, new capabilities |
| Bug fixes | PATCH | Fix bugs, security issues |
| Documentation | PATCH | Update docs, examples |
| Infrastructure | PATCH | CI/CD, Docker updates |

## Development Workflow Integration

### Before Starting Work
1. Check `TODO.md` for planned work items
2. Understand changelog requirements for your changes
3. Plan version increment if needed

### During Development
1. Keep notes of changes for changelog entry
2. Consider impact on users and versioning
3. Document breaking changes clearly

### Upon Completion
1. **Update CHANGELOG.md** with detailed entry
2. **Mark TODO.md items** as completed
3. **Cross-reference** between both files
4. **Commit both files** together with the feature
5. **Quality check** entry for completeness

### Example Workflow
```bash
# Start work
git checkout -b feature/new-algorithm

# Make changes
# ... development work ...

# Check if changelog update needed
git changelog-check

# Create changelog entry interactively
git changelog-update

# Commit everything together
git add CHANGELOG.md TODO.md src/
git commit -m "feat: add new anomaly detection algorithm

- Implement DBSCAN algorithm adapter
- Add comprehensive test coverage
- Update documentation with usage examples

Resolves #123"

# Push and create PR
git push origin feature/new-algorithm
```

## Best Practices

### Entry Quality
- **Be specific**: Describe what changed and why
- **Include context**: Help users understand impact
- **Reference issues**: Link to GitHub issues/PRs when relevant
- **Use active voice**: "Added feature X" vs "Feature X was added"
- **Group related changes**: Combine similar changes in one entry

### Version Management
- **Increment appropriately**: Don't skip versions
- **Date consistently**: Use YYYY-MM-DD format
- **Release regularly**: Don't accumulate too many changes
- **Tag releases**: Create git tags for versions

### Team Coordination
- **Review changelog entries**: Include in code review process
- **Discuss breaking changes**: Coordinate with team before major versions
- **Maintain consistency**: Follow established patterns
- **Update together**: Always commit changelog with features

## Troubleshooting

### Common Issues

**"Pre-commit hook blocking my commit"**
- Run `git changelog-check` to see what needs updating
- Use `git changelog-update` for interactive help
- Emergency bypass: `git commit --no-verify` (use sparingly)

**"Changelog entry format is wrong"**
- Follow the template in this guide
- Use the interactive helper for proper formatting
- Check existing entries for examples

**"Version number confusion"**
- Check current version: `git changelog-recent`
- Use semantic versioning guidelines
- When in doubt, use PATCH increment

**"Too many changes to document"**
- Break large changes into smaller commits
- Group related changes by category
- Focus on user-facing impacts

### Getting Help

1. **Read the rules**: Check `CLAUDE.md` > Changelog Management Rules
2. **Use interactive helper**: `python3 scripts/update_changelog_helper.py`
3. **Check existing entries**: Look at `CHANGELOG.md` for examples
4. **Ask the team**: Get help with version increments or entry content

## Automation Status

### ✅ Implemented
- [x] Changelog management rules in CLAUDE.md
- [x] Pre-commit hook validation
- [x] Interactive changelog helper
- [x] Git aliases for convenience
- [x] GitHub Actions PR validation
- [x] Pull request template integration
- [x] Installation automation script

### 🔄 Future Enhancements
- [ ] Release automation based on changelog
- [ ] Automatic version tagging
- [ ] Changelog-based release notes generation
- [ ] Integration with issue tracking systems
- [ ] Changelog preview in PRs

## Summary

The Pynomaly changelog automation system provides comprehensive tooling to ensure that all significant changes are properly documented. By combining validation scripts, interactive helpers, git integration, and CI/CD automation, the system makes it easy for developers to maintain high-quality version history while minimizing the burden of manual documentation.

The system strikes a balance between automation and human oversight, ensuring that changelog entries are both accurate and meaningful for users while being efficient for developers to maintain.

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
