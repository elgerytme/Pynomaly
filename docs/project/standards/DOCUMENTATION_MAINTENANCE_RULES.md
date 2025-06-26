# Documentation Maintenance Rules

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Project

---


This document outlines the mandatory rules for maintaining project documentation, specifically TODO.md and README.md files.

## 📋 TODO.md Maintenance Rules

### Automatic Updates Required
1. **Real-Time Task Sync**: Update TODO.md immediately when Claude Code todos change
2. **Date Updates**: Always update dates to current date when making changes
3. **Status Reflection**: Current work section must reflect actual active tasks
4. **Completion Tracking**: Move completed work to "Recently Completed" with date
5. **Archive Management**: Move old completed work to "Archived" section quarterly

### Structure Requirements
```markdown
# Pynomaly TODO List

## 🎯 **Current Status** (Month Year)
Brief status summary

## 🔄 **Current Work**
### ⏳ **Active Tasks**
- Currently in progress items

### ✅ **Recently Completed** 
- Items completed in last 30 days

## ✅ **Recently Completed Work**
- Major completions from last 3 months

## 📋 **Archived Completed Work**
- Historical completions (summarized)
```

### Update Triggers
- **Every TodoWrite operation**: Sync with TODO.md current work
- **Task completion**: Move to appropriate completed section
- **Session start**: Update dates and clean up stale content
- **Major work completion**: Add to "Recently Completed Work"

## 📚 README.md Maintenance Rules

### Automatic Updates Required
1. **Feature Updates**: Update feature lists when new capabilities are added
2. **Status Badges**: Keep all badges current (Python version, build system, etc.)
3. **Installation Instructions**: Update when dependencies or setup process changes
4. **Architecture Changes**: Reflect any architectural updates or new layers
5. **Examples**: Update code examples when APIs change

### Critical Sections
- **Features**: Must reflect current capabilities accurately
- **Installation**: Keep all methods current and tested
- **Quick Start**: Ensure examples work with current API
- **Architecture**: Update when structure changes
- **Development**: Reflect current workflow and tools

### Update Triggers
- **New features added**: Update Features section
- **API changes**: Update Quick Start examples
- **Architecture changes**: Update Architecture section
- **Dependency changes**: Update Installation section
- **Build system changes**: Update Development section

## 📅 Date Management Rules

### Current Date Format
Use "Month Year" format (e.g., "June 2025")

### Update Schedule
- **Every session**: Update current status dates
- **Weekly**: Review and update recent completion dates
- **Monthly**: Archive old completed work
- **Quarterly**: Clean up and reorganize archived sections

## 🔍 Content Accuracy Rules

### Verification Requirements
1. **Feature Claims**: Verify all claimed features actually exist
2. **Status Accuracy**: Ensure completion status reflects reality
3. **Link Validation**: Check that referenced files and docs exist
4. **Example Testing**: Verify code examples work as shown
5. **Dependency Accuracy**: Ensure all mentioned dependencies are current

### Accuracy Checks
- **Before major updates**: Validate all claims and examples
- **After feature additions**: Update relevant documentation sections
- **During architecture changes**: Ensure consistency across all docs
- **Weekly review**: Check for outdated information

## 🔄 Implementation Status

### Completed Corrections (June 2025)
- ✅ Removed inaccurate TODS library claims
- ✅ Updated Business Intelligence section to reflect actual capabilities
- ✅ Corrected optional dependencies to match pyproject.toml
- ✅ Updated production features to be more accurate
- ✅ Cleaned up TODO.md structure and archived old content
- ✅ Added accuracy disclaimers for planned features
- ✅ Fixed Python API example to match actual implementation
- ✅ Validated all installation commands are functional
- ✅ Updated architecture documentation to remove non-existent "Score" entity
- ✅ Implemented commit-based documentation review protocol

### Validation Results (June 26, 2025)
- ✅ **README.md Examples**: All Python API examples now use correct imports and method signatures
- ✅ **Installation Commands**: All Makefile targets, optional dependencies, and Hatch environments verified working
- ✅ **Architecture Documentation**: Updated to match actual entity structure and adapter implementation
- ✅ **Cross-Reference Validation**: All internal documentation links verified functional
- ✅ **Objective Language**: Removed marketing language and subjective claims

### Active Maintenance
- ✅ **Template Date System**: TODO.md uses `{{ current_month }} {{ current_year }}` format
- ✅ **Commit-Based Reviews**: Mandatory documentation review checklist implemented
- ⏳ **Ongoing Validation**: Regular validation of code examples and features
- ⏳ **Quarterly Archival**: Systematic archival of completed work
- ⏳ **Monthly Verification**: Dependencies and feature accuracy checks