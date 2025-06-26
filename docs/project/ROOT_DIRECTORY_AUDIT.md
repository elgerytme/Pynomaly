# Root Directory Audit - Organizational Violations

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Project

---


**Date**: June 26, 2025  
**Auditor**: Claude Code  
**Status**: Phase 1.1 Complete - Violations Identified

## 🎯 Audit Summary

**Current Root Directory Files**: 48+ files and directories  
**Target (Per Plan)**: ≤12 essential files  
**Violation Count**: 36+ items requiring relocation  
**Compliance Status**: ❌ **NON-COMPLIANT**

---

## ✅ **COMPLIANT FILES** (Keep in Root)

### Essential Project Files
- `README.md` ✅
- `LICENSE` ✅ 
- `CHANGELOG.md` ✅
- `TODO.md` ✅
- `CLAUDE.md` ✅

### Configuration Files
- `pyproject.toml` ✅
- `requirements.txt` ✅
- `package.json` ✅
- `package-lock.json` ✅

### Git Configuration
- `.gitignore` ✅ (hidden)
- `.gitattributes` ✅ (if exists, hidden)

### Development Tools
- `Makefile` ✅
- `Pynomaly.code-workspace` ✅

**Total Compliant**: 12 files ✅

---

## ❌ **VIOLATION CATEGORIES**

### 1. **Version Artifacts** (Delete)
- `2.0` ❌
- `=0.2.0.1` ❌
- `=0.46.0` ❌
- `=7.0.0` ❌

### 2. **Testing Files** (→ `tests/`)
- `test_core_functionality.py` ❌
- `test_setup.py` ❌
- `test_results_final.md` ❌
- `test_new_environment_comprehensive.ps1` ❌
- `test_new_environment_comprehensive.sh` ❌
- `test_powershell_simulation_comprehensive.sh` ❌
- `test_readme_cross_platform_comprehensive.sh` ❌
- `test_readme_instructions_bash.sh` ❌
- `test_readme_instructions_powershell.ps1` ❌
- `test_readme_instructions_powershell_simulation.sh` ❌

### 3. **Documentation Files** (→ `docs/`)
- `PACKAGE_FIX_SUMMARY.md` ❌
- `PROJECT_ORGANIZATION_PLAN.md` ❌ (should go to `docs/project/`)
- `README_INTEGRATION_TESTING.md` ❌
- `README_TESTING_REPORT.md` ❌
- `SCRIPT_TESTING_REPORT.md` ❌
- `TESTING_IMPROVEMENT_PLAN.md` ❌

### 4. **Build/Configuration Files** (→ `config/` or specific directories)
- `BUCK` ❌ (→ `deploy/build-configs/`)
- `MANIFEST.in` ❌ (→ `config/`)
- `playwright.config.ts` ❌ (→ `config/`)
- `pytest-bdd.ini` ❌ (→ `config/`)
- `pytest.ini` ❌ (→ `config/`)
- `lighthouse.config.js` ❌ (→ `config/`)
- `lighthouserc.js` ❌ (→ `config/`)
- `tailwind.config.js` ❌ (→ `config/web/`)
- `tox.ini` ❌ (→ `config/`)

### 5. **Scripts** (→ `scripts/`)
- `execute_cli_testing_plan.sh` ❌
- `find_real_errors.py` ❌
- `find_undefined_names.py` ❌
- `fix_package_issues.py` ❌
- `fix_windows_setup.ps1` ❌
- `setup.bat` ❌

### 6. **Data/Storage Files** (→ `storage/` or gitignore)
- `tdd_config.json` ❌ (→ `config/`)
- `TODO.md.backup` ❌ (delete)
- `advanced_testing_config.json` ❌ (→ `config/`)

### 7. **Build Artifacts** (→ `reports/builds/` or gitignore)
- `dist/` ❌
- `buck-out/` ❌
- JSON reports in root (→ `reports/`)

### 8. **Requirements Sprawl** (→ `config/environments/`)
- `requirements-minimal.txt` ❌
- `requirements-production.txt` ❌
- `requirements-server.txt` ❌
- `requirements-test.txt` ❌

### 9. **Node.js Artifacts** (Keep but ensure gitignored)
- `node_modules/` ❌ (should be gitignored)

---

## 📋 **RELOCATION PLAN**

### **Phase 1A: Delete Version Artifacts**
```bash
rm -f "2.0" "=0.2.0.1" "=0.46.0" "=7.0.0"
rm -f "TODO.md.backup"
```

### **Phase 1B: Move Testing Files**
```bash
mv test_*.py tests/
mv test_*.sh tests/scripts/
mv test_*.ps1 tests/scripts/
mv *_TESTING_*.md docs/testing/
```

### **Phase 1C: Move Documentation**
```bash
mv PROJECT_ORGANIZATION_PLAN.md docs/project/
mv *_SUMMARY.md docs/project/
mv *_REPORT.md docs/project/
mv TESTING_IMPROVEMENT_PLAN.md docs/project/plans/
```

### **Phase 1D: Move Configuration Files**
```bash
mv BUCK deploy/build-configs/
mv MANIFEST.in config/
mv *.ini config/
mv *.config.* config/
mv lighthouserc.js config/web/
```

### **Phase 1E: Move Scripts**
```bash
mv *.py scripts/maintenance/
mv *.sh scripts/testing/
mv *.ps1 scripts/testing/
mv setup.bat scripts/setup/
```

### **Phase 1F: Move Requirements Files**
```bash
mkdir -p config/environments/
mv requirements-*.txt config/environments/
```

---

## 🎯 **POST-CLEANUP TARGET STATE**

**Root Directory (12 files max)**:
```
pynomaly/
├── README.md
├── LICENSE  
├── CHANGELOG.md
├── TODO.md
├── CLAUDE.md
├── pyproject.toml
├── requirements.txt
├── package.json
├── package-lock.json
├── Makefile
├── Pynomaly.code-workspace
└── .gitignore
```

---

## 🔍 **ENFORCEMENT REQUIREMENTS**

### **Immediate Actions**
1. ✅ Complete this audit
2. ⏳ Create relocation scripts
3. ⏳ Execute file moves
4. ⏳ Update .gitignore
5. ⏳ Implement pre-commit hooks

### **Validation Checks**
- Root directory file count ≤ 12
- No testing files in root
- No documentation in root
- No build artifacts in root
- No script files in root

### **Automation Hooks**
- Pre-commit validation
- CI/CD compliance checking
- Automated violation reporting

---

**Next Step**: Proceed to Phase 1.2 - Execute file relocations with proper directory creation and safety checks.