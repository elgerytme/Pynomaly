# Container Migration Audit

This document provides a checklist of files and sections within the Pynomaly repository that should be updated or deprecated in favor of container usage.

### Files/Sections to Update or Deprecate

1. **README.md**
   - Update virtual environment instructions to recommend Docker usage for consistent environments across different systems.

2. **pyproject.toml**
   - Consider removing or updating environment-specific settings in favor of consistent Docker environments.

3. **Makefile**
   - Simplify or replace virtual environment targets with Docker-based workflows.

4. **.github/workflows Directory**
   - Review CI configurations to favor Docker containers for build and test environments to avoid discrepancies between local and CI environments.

5. **scripts Directory**
   - Review and potentially deprecate scripts related to setting up Python virtual environments (e.g., `setup_simple.py`, `setup_fresh_environment.sh`).

6. **requirements.txt and Related Files**
   - Ensure Dockerfiles are used primarily for dependency management to ensure uniformity across environments.

7. **config/.env.example**
   - Encourage using environment variables within Docker configurations for consistency.

8. **deploy/docker Directory**
   - Ensure Docker configurations are central and comprehensive, possibly deprecating non-Docker setup instructions.

9. **docs/getting-started/installation.md**
   - Highlight Docker as the primary method of setting up the application.

10. **docs/getting-started/FEATURE_INSTALLATION_GUIDE.md**
    - Suggest Docker images for different feature sets rather than individual Python package installations.

### Docker Configuration and Migration Plan
- Use existing Docker configurations as a baseline to ensure uniformity across development, CI, and production environments.
- Consider updating `docker-compose.yml` and `Dockerfile` to cover all aspects of deployment, from local development to production.
- Align existing `.env` configuration with Docker's environment variable usage.

Transitioning to containers will streamline the setup and deployment processes, ensuring a standardized environment across all platforms.
