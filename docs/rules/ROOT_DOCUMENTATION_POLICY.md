# Root Documentation Policy

This document defines the policy for documentation files in the repository root directory.

## Policy Statement

**No new documentation files may be created or moved into the repository root directory.**

The repository root should remain clean and focused on essential project files only.

## Allowed Root Documentation Files

Only the following documentation files are permitted in the repository root:

### Core Project Files
- `README.md` - Main project overview and getting started guide
- `CHANGELOG.md` - Version history and release notes  
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - Software license
- `NOTICE` - Copyright and attribution notices
- `REPOSITORY_RULES.md` - Repository governance and development rules

## Grandfathered Files

The following files existed before this policy was implemented and are temporarily allowed:

⚠️ **These should eventually be moved to appropriate locations:**

- `COMPREHENSIVE_TEST_COMPLETENESS_REPORT.md` → `docs/reports/`
- `PACKAGE_VALIDATION_REPORT.md` → `docs/reports/`
- `TEST_IMPROVEMENT_RECOMMENDATIONS.md` → `docs/reports/`

## Proper Locations for Documentation

### Repository Documentation Structure
```
docs/
├── reports/              # Analysis reports, summaries, assessments
├── guides/               # User guides, tutorials, how-to documents
├── architecture/         # Architecture decisions, design documents  
├── rules/                # Governance, policies, standards
├── security/             # Security policies, procedures
├── development/          # Development guides, workflows
├── deployment/           # Deployment guides, procedures
└── examples/             # Code examples, samples
```

### Package Documentation Structure
```
src/packages/{domain}/{package}/
├── docs/
│   ├── reports/          # Package-specific reports
│   ├── guides/           # Package usage guides
│   ├── architecture/     # Package design docs
│   └── examples/         # Package examples
└── README.md             # Package overview (allowed)
```

## Enforcement

This policy is enforced through:

### 1. Pre-commit Hook
Automatically validates on every commit:
```bash
# Runs automatically on git commit
python tools/scripts/validate_root_documentation.py --suggest-moves
```

### 2. Buck2 Build Validation  
Integrated into build system:
```bash
# Standard validation
buck2 run //:validate-root-documentation

# Strict mode (fails on grandfathered files too)
buck2 run //:validate-root-documentation-strict
```

### 3. Manual Validation
Run validation manually:
```bash
# Basic check with suggestions
python tools/scripts/validate_root_documentation.py --suggest-moves

# Strict check (fails on all root docs except allowed)
python tools/scripts/validate_root_documentation.py --fail-on-grandfathered --suggest-moves
```

## File Classification

The validation system identifies documentation files based on:

### File Extensions
- `.md` (Markdown)
- `.txt` (Plain text)
- `.rst` (reStructuredText)
- `.adoc` (AsciiDoc)
- `.pdf` (PDF documents)

### File Names
Files without extensions that are typically documentation:
- `README`, `LICENSE`, `NOTICE`, `CHANGELOG`, `CONTRIBUTING`, `AUTHORS`, `CREDITS`

### Content Patterns
Files with documentation-style naming:
- Contains: `readme`, `guide`, `tutorial`, `howto`, `faq`, `manual`, `docs`, `spec`

## Moving Existing Files

If you need to move a documentation file from the root:

### 1. Determine Appropriate Location
Use the validation script to get suggestions:
```bash
python tools/scripts/validate_root_documentation.py --suggest-moves
```

### 2. Move the File
```bash
# Example: Moving a report
mv SOME_REPORT.md docs/reports/

# Example: Moving a guide  
mv SETUP_GUIDE.md docs/guides/
```

### 3. Update References
Search for and update any references to the moved file:
```bash
# Find references
grep -r "SOME_REPORT.md" .

# Update links in other documentation
# Update any scripts or automation that references the file
```

## Rationale

This policy exists to:

1. **Maintain Clean Root Directory** - Keep the repository root focused on essential files
2. **Improve Organization** - Group related documentation in appropriate subdirectories
3. **Enhance Discoverability** - Make it easier to find documentation by category
4. **Prevent Documentation Sprawl** - Avoid accumulation of miscellaneous docs in root
5. **Standardize Structure** - Create consistent expectations across the repository

## Examples

### ✅ Allowed Patterns

```
# Core project files
README.md
LICENSE
CONTRIBUTING.md

# Organized documentation
docs/reports/ANALYSIS_REPORT.md
docs/guides/GETTING_STARTED.md
src/packages/data/analytics/docs/guides/USAGE.md
```

### ❌ Prohibited Patterns

```
# New documentation files in root
DEPLOYMENT_GUIDE.md          # Should be docs/deployment/
API_DOCUMENTATION.md         # Should be docs/guides/
SECURITY_POLICY.md           # Should be docs/security/
ANALYSIS_RESULTS.md          # Should be docs/reports/
PACKAGE_OVERVIEW.md          # Should be in package docs/
```

## Exceptions Process

In rare cases where an exception might be warranted:

1. **Create Issue** - Document the specific need and rationale
2. **Team Discussion** - Discuss with repository maintainers
3. **Update Policy** - If approved, update this policy document
4. **Update Whitelist** - Modify `ALLOWED_ROOT_DOCS` in validation script

## Migration Timeline

For grandfathered files:

- **Phase 1** (Current): Warning mode - files flagged but allowed
- **Phase 2** (Future): Consider moving grandfathered files to proper locations
- **Phase 3** (Future): Enable strict mode to prevent all non-whitelisted root documentation

---

**Policy Version:** 1.0  
**Effective Date:** 2025-01-22  
**Last Updated:** 2025-01-22  
**Maintained By:** Repository Governance Team