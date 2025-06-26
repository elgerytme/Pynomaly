# Commit-Based Documentation Rules

This document outlines the mandatory rules for maintaining objective, accurate documentation that must be reviewed and updated with every commit.

## 🎯 **Objective Documentation Principle**

**README.md must ONLY contain objective, verifiable information about the actual status and features of the project. No subjective claims, marketing language, or aspirational content is permitted.**

## 📋 **Mandatory Pre-Commit Documentation Review**

### 🔄 **Every Commit Must Include:**

1. **README.md Verification**
   - ✅ All features listed are actually implemented and testable
   - ✅ All installation commands work as documented
   - ✅ All code examples execute successfully
   - ✅ All dependencies match actual pyproject.toml
   - ✅ Architecture section matches actual codebase structure
   - ✅ No marketing language or subjective claims

2. **TODO.md Synchronization**
   - ✅ Current work reflects actual Claude Code todos
   - ✅ Date templates are resolved to current date
   - ✅ Completion status is accurate and verifiable
   - ✅ Recent completions are properly timestamped
   - ✅ Archived work is appropriately organized

3. **Cross-Reference Validation**
   - ✅ Consistency between README.md and TODO.md
   - ✅ All referenced files and directories exist
   - ✅ All documentation links are functional
   - ✅ Examples match current API implementation

## 📝 **TODO.md Template System**

### Date Template Format
```markdown
## 🎯 **Current Status** (June 2025)
<!-- Template: {{ current_month }} {{ current_year }} -->
```

### Automatic Resolution
- `{{ current_month }}` → Current month name (e.g., "June")
- `{{ current_year }}` → Current year (e.g., "2025")
- Templates must be resolved to actual dates before commits

### Structure Requirements
- **Current Status**: Brief, objective summary of project state
- **Current Work**: Active tasks from Claude Code todos
- **Recently Completed**: Last 30 days with dates
- **Recently Completed Work**: Last 3 months with context
- **Archived Completed Work**: Historical summaries

## 🚫 **Prohibited Content in README.md**

### ❌ Marketing Language
- "State-of-the-art"
- "Revolutionary"
- "Cutting-edge"
- "Modern" (unless technically specific)
- "Advanced" (unless technically specific)
- "Lightning-fast"
- "Comprehensive" (unless verifiable)

### ❌ Aspirational Claims
- Features not yet implemented
- "Planned" features without clear disclaimers
- "Upcoming" capabilities
- Future roadmap items
- Unverified performance claims

### ❌ Unverified Features
- Algorithm support without working adapters
- Integration claims without actual implementations
- Performance metrics without benchmarks
- Dependencies not in pyproject.toml

## ✅ **Required Content Standards**

### ✅ Objective Language
- Factual descriptions of implemented features
- Technical specifications with versions
- Measurable performance characteristics
- Verifiable architecture components

### ✅ Testable Claims
- All installation methods must work
- All code examples must execute
- All commands must be functional
- All dependencies must be current

### ✅ Current Status
- Features reflect actual codebase capabilities
- Architecture matches directory structure
- Dependencies match configuration files
- Examples use current API

## 🔍 **Verification Checklist**

### Before Every Commit:
- [ ] **README.md Features**: Each feature is verifiable in codebase
- [ ] **README.md Examples**: All code examples execute successfully
- [ ] **README.md Installation**: All commands work as documented
- [ ] **README.md Dependencies**: Match actual pyproject.toml
- [ ] **README.md Architecture**: Reflects actual structure
- [ ] **README.md Language**: No marketing or subjective terms
- [ ] **TODO.md Status**: Reflects actual work and completion state
- [ ] **TODO.md Dates**: Templates resolved to current date
- [ ] **TODO.md Accuracy**: All claims are verifiable
- [ ] **Cross-References**: All links and file references work
- [ ] **Consistency**: README.md and TODO.md are aligned

## 🚨 **Enforcement Rules**

1. **No Commit Without Review**: Documentation review is mandatory before any commit
2. **Accuracy Over Marketing**: Objective truth always takes precedence
3. **Current State Only**: Document what exists, not what's planned
4. **Verifiable Claims**: Every claim must be testable
5. **Template Resolution**: Date templates must be current

## 📅 **Update Schedule**

- **Every Commit**: Complete documentation review and updates
- **Every Session**: Resolve date templates and verify accuracy
- **Weekly**: Review recent completions and archive old items
- **Monthly**: Archive completed work and update organization
- **Quarterly**: Major reorganization and historical archiving

## 🎯 **Success Criteria**

Documentation is compliant when:
- ✅ All README.md claims are objectively verifiable
- ✅ All TODO.md status reflects actual work state
- ✅ All examples and commands work as documented
- ✅ No marketing language or aspirational content remains
- ✅ All dates are current and accurate
- ✅ All cross-references are functional
- ✅ Architecture documentation matches codebase reality