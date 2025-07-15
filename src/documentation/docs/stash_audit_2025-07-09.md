# Stash Audit - 2025-07-09

This document tracks the analysis of each stash processed on 2025-07-09.

## Processing Status
- **CANDIDATE**: Stash contains relevant changes and should be reviewed for manual processing
- **SKIPPED**: Stash contains no relevant file changes
- **FAILED**: Technical error during processing
- **COMMITTED**: Stash successfully applied and committed to feature branch
- **DROPPED**: Stash failed tests or contained irrelevant changes

| Index | Hash    | Status  | Notes |
|-------|---------|---------|-------|
| 0 | 03b5e426 | CANDIDATE | Relevant: 1473 files, 2020 files changed, 1942096 insertions(+), 1942168 deletions(-) |
| 1 | f260de89 | CANDIDATE | Relevant: 2 files, 242 files changed, 24 insertions(+), 96 deletions(-) |
| 2 | c472efd7 | CANDIDATE | Relevant: 8 files, 787 files changed, 246 insertions(+), 249 deletions(-) |
| 3 | 504daee6 | CANDIDATE | Relevant: 12 files, 346 files changed, 7494 insertions(+), 12008 deletions(-) |
| 4 | 2ffccd57 | CANDIDATE | Relevant: 1 file, 1 file changed, 9 insertions(+), 19 deletions(-) |
| 5 | 60ce3621 | CANDIDATE | Relevant: 2 files, 2 files changed, 12 insertions(+), 25 deletions(-) |
| 6 | 058520e2 | CANDIDATE | Relevant: 2 files, 20 files changed, 20 insertions(+), 9 deletions(-) |
| 7 | 9e0de410 | CANDIDATE | Relevant: 4 files, 16 files changed, 815 insertions(+), 204 deletions(-) |
| 8 | a8dabc71 | CANDIDATE | Relevant: 4 files, 6 files changed, 703 insertions(+) |
| 9 | d72b8e85 | CANDIDATE | Relevant: 6 files, 9 files changed, 342 insertions(+), 195 deletions(-) |
| 10 | 136c93e7 | CANDIDATE | Relevant: 314 files, 631 files changed, 3140 insertions(+), 161575 deletions(-) |
| 11 | a4d0c72e | CANDIDATE | Relevant: 67 files, 147 files changed, 21 insertions(+), 37580 deletions(-) |
| 12 | a6354103 | CANDIDATE | Relevant: 133 files, 416 files changed, 76 insertions(+), 76179 deletions(-) |
| 13 | e2c73177 | CANDIDATE | Relevant: 1 file, 817 files changed, 4 insertions(+), 2 deletions(-) |
| 14 | 4bd543b6 | CANDIDATE | Relevant: 4 files, 6 files changed, 377 insertions(+), 3 deletions(-) |
| 15 | 5ebbd7a5 | CANDIDATE | Relevant: 13 files, 13 files changed, 79 insertions(+), 930 deletions(-) |
| 16 | 16dfb24a | CANDIDATE | Relevant: 1 file, 2 files changed, 73 insertions(+), 9 deletions(-) |

## Summary

**Total Stashes Processed:** 17  
**Candidates for Manual Review:** 17  
**Skipped (No Relevant Changes):** 0  
**Failed:** 0  

## Recommendations

1. **High Priority Stashes (Large Change Sets):**
   - Stash 0 (03b5e426): 1473 files - Major cleanup/refactoring
   - Stash 10 (136c93e7): 314 files - PWA/UX completion work
   - Stash 11 (a4d0c72e): 67 files - Access request model changes
   - Stash 12 (a6354103): 133 files - Multiple feature branch changes

2. **Medium Priority Stashes (Moderate Changes):**
   - Stash 3 (504daee6): 12 files - CI/Database test changes
   - Stash 15 (5ebbd7a5): 13 files - Pytest configuration changes
   - Stash 2 (c472efd7): 8 files - AutoML/PyOD adapter changes

3. **Low Priority Stashes (Small Changes):**
   - Stash 4 (2ffccd57): 1 file - Anomaly domain entity changes
   - Stash 13 (e2c73177): 1 file - Comprehensive testing workflow
   - Stash 16 (16dfb24a): 1 file - Database persistence changes

## Next Steps

1. Review high priority stashes manually
2. Create feature branches for relevant changes
3. Run tests on applied changes
4. Create PRs for validated stashes
5. Drop stashes that fail tests or are no longer relevant

---
*Audit completed on 2025-07-09 using automated stash analysis*

