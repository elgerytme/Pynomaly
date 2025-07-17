# Replacement Folders and Files Issue

## Problem Statement

The repository has been experiencing a recurring issue where replacement folders and files are being created when the original files cannot be found. This creates a cycle of:

1. Files/folders are removed or relocated
2. Systems or processes try to access these files/folders
3. When not found, replacement files/folders are automatically created
4. This leads to architectural drift and the reintroduction of previously removed structures

## Specific Examples

### Core Folders Issue
- **Issue**: "Core" folders were removed from the repository to eliminate generic naming
- **Problem**: Systems kept recreating "core" folders when they couldn't find the original ones
- **Impact**: Architecture degradation, reintroduction of anti-patterns

### Pattern Observed
1. Remove deprecated/problematic structure
2. System tries to access removed structure
3. System creates replacement structure when not found
4. Problem resurfaces with new structure

## Root Causes

1. **Automatic Folder Creation**: Systems creating folders when they can't find expected paths
2. **Hardcoded Paths**: Code references to specific folder structures that no longer exist
3. **Lack of Validation**: No checks to prevent creation of prohibited structures
4. **Missing Migration**: Old references not updated when structures are moved/removed

## Impact on Repository

- **Architectural Drift**: Slow degradation of intended architecture
- **Repeated Work**: Same issues fixed multiple times
- **Confusion**: Unclear which structures are current vs. deprecated
- **Maintenance Burden**: Increased effort to maintain clean architecture

## Solution Requirements

1. **Repository Rules**: Explicit rules about prohibited folder/file patterns
2. **Validation**: Automated checks to prevent creation of prohibited structures
3. **Documentation**: Clear guidelines on what should/shouldn't be created
4. **Migration Support**: Proper handling of moved/removed structures
5. **Monitoring**: Detection of unwanted structure creation

## Repository Rule Implementation

See: `REPOSITORY_RULES.md` for the formal rule implementation.

## GitHub Issue Tracking

GitHub Issue: https://github.com/elgerytme/Pynomaly/issues/830

## Resolution Steps

1. ✅ Document the issue (this document)
2. ✅ Create repository rule
3. ✅ Create GitHub issue
4. ✅ Implement validation safeguards
5. ⏳ Add monitoring for prohibited structures

## Prevention Strategy

- **Pre-commit hooks**: Validate against prohibited patterns
- **CI/CD checks**: Automated validation in build pipeline
- **Documentation**: Clear guidelines for contributors
- **Code reviews**: Manual validation of structural changes