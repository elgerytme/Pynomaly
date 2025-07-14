# Documentation Standards

## Date Formatting Guidelines

### Overview

To ensure consistency and accuracy across all documentation, this guide establishes standards for date formatting and timezone handling.

### Date Format Standards

#### 1. **Primary Format: Full Month Name**
```markdown
**Last Updated**: July 11, 2025
**Created**: January 15, 2025
```

#### 2. **Alternative Format: Abbreviated Month**
```markdown
**Date**: Jul 11, 2025
**Modified**: Jan 15, 2025
```

#### 3. **ISO Format (for technical documentation)**
```markdown
**Last Updated**: 2025-07-11
**Created**: 2025-01-15
```

### Timezone Requirements

1. **Always use local timezone** for documentation dates
2. **Never use dates that are off by days, months, or years**
3. **Update dates when modifying documentation**

### Validation Rules

The automated date validation system enforces these rules:

- **Reasonable Range**: Dates must be within 30 days of current date
- **Format Consistency**: Use one of the approved formats consistently
- **Timezone Accuracy**: Dates reflect the actual local timezone

### Common Date Patterns

#### Markdown Headers
```markdown
**Last Updated**: July 11, 2025
**Date**: July 11, 2025
**Created**: July 11, 2025
**Modified**: July 11, 2025
```

#### YAML Frontmatter
```yaml
---
date: 2025-07-11
updated: 2025-07-11
created: 2025-01-15
---
```

#### Comments
```markdown
<!-- Last updated: July 11, 2025 -->
# Date: July 11, 2025
```

### Automation

#### Pre-commit Hook
The repository includes a pre-commit hook that validates dates before commits:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run validate-documentation-dates --all-files
```

#### GitHub Actions
Automatic validation runs on:
- Push to any branch (documentation files)
- Pull requests (documentation files) 
- Daily at 9:00 AM UTC (to catch date drift)
- Manual workflow dispatch

#### Manual Validation
```bash
# Check dates only
python scripts/automation/validate_dates.py

# Fix incorrect dates automatically
python scripts/automation/validate_dates.py --fix
```

### File Types Covered

The validation system checks these file types:
- `*.md` (Markdown files)
- `*.yml` and `*.yaml` (YAML files)
- Documentation in `docs/**` directory
- README files
- Workflow files in `.github/workflows/`

### Error Examples

#### ❌ Incorrect Dates
```markdown
**Last Updated**: January 1, 2020  <!-- Too far in the past -->
**Date**: December 25, 2030        <!-- Too far in the future -->
**Created**: February 30, 2025     <!-- Invalid date -->
```

#### ✅ Correct Dates
```markdown
**Last Updated**: July 11, 2025    <!-- Current or recent date -->
**Date**: July 1, 2025             <!-- Within reasonable range -->
**Created**: June 15, 2025         <!-- Recent past date -->
```

### Best Practices

1. **Update dates when editing**: Always update the "Last Updated" field when making changes
2. **Use consistent formatting**: Stick to one format throughout a document
3. **Check timezone**: Ensure your system timezone is correct
4. **Run validation locally**: Use the pre-commit hook to catch issues early
5. **Review automation feedback**: Pay attention to validation warnings in PRs

### Troubleshooting

#### Date Validation Failures
If the validation system reports date issues:

1. **Check the reported file and line number**
2. **Verify the date is current and reasonable**
3. **Use the auto-fix option**: `python scripts/automation/validate_dates.py --fix`
4. **Commit the fixed dates**

#### Common Issues
- **System timezone incorrect**: Check your OS timezone settings
- **Copy-paste from old documents**: Always update dates when reusing content
- **Template files**: Ensure template dates are updated when used

### Implementation Details

#### Date Patterns Detected
The validation system recognizes these patterns:
- Full month names: "January 1, 2025"
- Abbreviated months: "Jan 1, 2025"  
- ISO format: "2025-01-01"
- Various separators: "01/01/2025", "01-01-2025"
- Markdown bold formatting: "**Date**: January 1, 2025"
- YAML fields: "date: 2025-01-01"
- Comments: "# Last updated: January 1, 2025"

#### Tolerance Window
- **±30 days** from current date is considered reasonable
- Dates outside this window trigger validation errors
- Future dates beyond 30 days are flagged
- Past dates older than 30 days are flagged

### Related Files

- **Validation Script**: `scripts/automation/validate_dates.py`
- **GitHub Workflow**: `.github/workflows/date-validation.yml`
- **Pre-commit Config**: `.pre-commit-config.yaml`
- **Pre-commit Hooks**: `.pre-commit-hooks.yaml`

---

**Last Updated**: July 11, 2025  
**Status**: Active and enforced via automation  
**Maintainer**: Automated CI/CD system