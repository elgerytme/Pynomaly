# Conflict Resolution Guide

## Rebase vs Merge Policy

- **Rebase**: Use rebase for personal feature branches that haven’t been shared with others. This helps maintain a clean, linear commit history.
  - Preferred when:
    - Working on feature branches that are not shared.
    - Before merging into `develop`, to clean up commit history.
  
- **Merge**: Use merge to integrate changes from shared branches with multiple contributors. Preserves the history and branch context.
  - Preferred when:
    - Working on shared branches.
    - Collaboration with multiple contributors is involved.

## How to Resolve Generated Artifacts & Lock Files

1. **Generated Artifacts**: These should not be committed to the repository. Use a `.gitignore` file to exclude any generated files.

2. **Lock Files**: When resolving conflicts in lock files (like `package-lock.json` or `yarn.lock`), regenerate them after the conflict is resolved:
   
   ```bash
   # Resolve conflicts manually in dependencies
   git add package-lock.json
   git rebase --continue

   # Regenerate lock file
   npm install
   
   # Check changes
   git diff
   ```

## Large-file Handling and `.gitattributes` Pointers

- **Large Files**: Use Git LFS (Large File Storage) for managing large files that shouldn’t be in the main Git history.
  
- **`.gitattributes`**: Configure `.gitattributes` to handle large binaries or specific file types that require different handling during merges.

  Example:
  ```
  *.psd filter=lfs diff=lfs merge=lfs -text
  ```
  
  Add the pointers in `.gitattributes` to track these files without bloating the Git history.

## Helper Makefile Target

For resolving conflicts, if available, use the Makefile target:
```bash
make fix-conflicts
```
This will aid in streamlining the conflict resolution process by efficiently managing the different targets and potential automation in the resolution scripts.
