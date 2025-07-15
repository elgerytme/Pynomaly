# Pull Request

## Description
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Check the type of change that applies to this PR -->

- [ ] 🐛 **Bug fix** (non-breaking change which fixes an issue)
- [ ] ✨ **New feature** (non-breaking change which adds functionality)
- [ ] 💥 **Breaking change** (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 **Documentation** (changes to documentation only)
- [ ] 🔧 **Refactoring** (code changes that neither fix a bug nor add a feature)
- [ ] ⚡ **Performance** (code changes that improve performance)
- [ ] 🧪 **Tests** (adding missing tests or correcting existing tests)
- [ ] 🏗️ **Infrastructure** (changes to CI/CD, build system, or deployment)
- [ ] 🔒 **Security** (security-related changes or vulnerability fixes)

## Changes Made
<!-- List the specific changes made in this PR -->

### Added
-

### Changed
-

### Fixed
-

### Removed
-

## Testing
<!-- Describe the tests you ran to verify your changes -->

- [ ] **Unit tests** pass
- [ ] **Integration tests** pass  
- [ ] **Manual testing** completed
- [ ] **Performance testing** (if applicable)
- [ ] **Security testing** (if applicable)

### Test Coverage
- [ ] **Existing tests** cover the changes
- [ ] **New tests** added for new functionality
- [ ] **Test coverage** maintained or improved

## Documentation
<!-- Check all documentation that has been updated -->

- [ ] **Code comments** updated
- [ ] **API documentation** updated
- [ ] **User documentation** updated
- [ ] **README.md** updated (if applicable)
- [ ] **Examples** updated (if applicable)

## Changelog Requirements ⚠️
<!-- This section is MANDATORY for all significant changes -->

### Changelog Update Status
- [ ] **CHANGELOG.md updated** with this change
- [ ] **Version increment** is appropriate (patch/minor/major)
- [ ] **Entry follows** the project's changelog format
- [ ] **TODO.md updated** to mark completed items (if applicable)

### Change Significance Assessment
<!-- Check if your changes meet any of these criteria requiring changelog update -->

- [ ] New features, capabilities, or functionality
- [ ] Changes in existing functionality or behavior  
- [ ] Bug fixes affecting functionality
- [ ] API changes (breaking or non-breaking)
- [ ] Security-related changes
- [ ] Performance improvements
- [ ] Infrastructure or deployment changes
- [ ] Documentation additions or major updates
- [ ] Algorithm implementations or ML changes
- [ ] New datasets, examples, or analysis tools
- [ ] 20+ lines of code changed in critical paths

**If ANY boxes above are checked, CHANGELOG.md update is REQUIRED**

### Changelog Entry Preview
<!-- If you updated CHANGELOG.md, paste the relevant section here -->

```markdown
## [X.Y.Z] - YYYY-MM-DD

### [Category]
- [Your changelog entry here]
```

## Dependencies
<!-- List any new dependencies or version changes -->

- [ ] **No new dependencies** added
- [ ] **New dependencies** documented in requirements
- [ ] **Dependency versions** updated appropriately
- [ ] **Optional dependencies** properly configured

## Breaking Changes
<!-- If this is a breaking change, describe the impact and migration path -->

- [ ] **No breaking changes**
- [ ] **Breaking changes documented** with migration guide
- [ ] **Deprecation warnings** added for removed functionality
- [ ] **Version bump** reflects breaking nature of changes

## Checklist
<!-- Ensure all items are completed before requesting review -->

### Code Quality
- [ ] **Code follows** project style guidelines
- [ ] **Self-review** completed
- [ ] **Comments added** for hard-to-understand areas
- [ ] **No commented-out code** or debug statements
- [ ] **Error handling** implemented appropriately

### Clean Architecture Compliance
- [ ] **Domain layer** remains dependency-free
- [ ] **Application layer** orchestrates use cases properly
- [ ] **Infrastructure layer** implements interfaces correctly
- [ ] **Presentation layer** handles user interactions appropriately
- [ ] **Dependency injection** used properly

### Production Readiness
- [ ] **Error handling** for edge cases
- [ ] **Logging** added for debugging
- [ ] **Performance** considerations addressed
- [ ] **Security** implications considered
- [ ] **Monitoring** hooks added (if applicable)

### Final Verification
- [ ] **All tests pass** locally
- [ ] **No merge conflicts** with target branch
- [ ] **Commit messages** follow conventional format
- [ ] **PR title** clearly describes the change
- [ ] **Reviewers assigned** and notified

## Additional Notes
<!-- Add any additional context, screenshots, or information that would be helpful for reviewers -->

---

## For Reviewers

### Review Focus Areas
<!-- Highlight specific areas that need careful review -->

- [ ] **Algorithm correctness** (for ML changes)
- [ ] **Security implications** (for security-related changes)
- [ ] **Performance impact** (for performance changes)
- [ ] **API design** (for interface changes)
- [ ] **Documentation accuracy** (for user-facing changes)

### Deployment Considerations
- [ ] **Database migrations** required
- [ ] **Configuration changes** needed
- [ ] **Infrastructure updates** required
- [ ] **Feature flags** needed for rollout

---

**Note**: This PR template enforces the changelog update rule established in CLAUDE.md. All significant changes must include a CHANGELOG.md update to maintain proper version history and release notes.
