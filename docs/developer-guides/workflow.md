# End-to-End Validation Checklist

🍞 **Breadcrumb:** 🏠 [Home](../index.md)  👨‍💻 [Developer Guides](./README.md)  🔄 Workflow

---

The following steps outline the process to ensure your environment and codebase remain consistent and error-free.

1. **Setup Project**
   - Run `make setup` or use `hatch` commands as required. This should complete with no errors.
   
2. **Code Quality**
   - Run `make lint` or relevant `hatch` command.
   - Ensure all code quality checks pass.

3. **Testing**
   - Run `make test` or `hatch` command equivalent.
   - Verify that all tests pass and a coverage report is generated.

4. **Documentation**
   - Execute `make docs` to build documentation.
   - Ensure the documentation site loads without broken links.

5. **Pre-Commit**
   - Make a commit.
   - Verify that pre-commit hooks trigger automatically with no failures.
