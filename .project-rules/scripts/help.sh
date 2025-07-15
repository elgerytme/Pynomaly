#!/bin/bash

# Pynomaly Isolation Help Script
# Provides context-aware help for isolation environment

cat << 'EOF'
🔒 Pynomaly Isolation Environment Help
====================================

📁 Current Environment Information:
  - Workspace: /workspace
  - Python: $(python --version 2>/dev/null || echo "Not available")
  - Git Branch: $(git branch --show-current 2>/dev/null || echo "No Git repository")
  - Isolation Mode: ${ISOLATION_MODE:-Unknown}
  - Environment: ${PYNOMALY_ENV:-Unknown}

🛠️  Available Commands:
===================

Development Commands:
  ./isolation/scripts/start-dev.sh    Start development server
  ./isolation/scripts/test.sh         Run test suite
  ./isolation/scripts/lint.sh         Run code linting
  ./isolation/scripts/format.sh       Format code
  ./isolation/scripts/type-check.sh   Run type checking

Database Commands:
  ./isolation/scripts/db-setup.sh     Initialize database
  ./isolation/scripts/db-reset.sh     Reset database
  ./isolation/scripts/db-migrate.sh   Run migrations

Git Commands:
  git status                          Check repository status
  git branch                          List branches
  git checkout -b feature/my-feature  Create new feature branch
  git add .                           Stage changes
  git commit -m "message"             Commit changes

Python/Pip Commands:
  pip install package                 Install Python package
  pip list                           List installed packages
  pip freeze > requirements.txt      Export requirements
  python -m pytest                   Run tests manually
  python -c "import sys; print(sys.path)"  Check Python path

📊 Useful Aliases and Functions:
==============================
  ll          - Detailed file listing
  la          - List all files including hidden
  ..          - Go up one directory
  ...         - Go up two directories

🔧 Isolation-Specific Information:
================================
  - All changes are isolated from the main repository
  - Use regular Git commands to manage your changes
  - Exit the container to return to the main environment
  - Changes will persist in the isolation until cleanup

🚀 Quick Start Workflow:
======================
  1. ./isolation/scripts/start-dev.sh    # Start development server
  2. # Make your changes in /workspace
  3. ./isolation/scripts/test.sh         # Run tests
  4. git add . && git commit -m "..."    # Commit changes
  5. exit                                # Leave isolation

📋 Environment Variables:
=======================
  PYTHONPATH=/workspace/src
  PYNOMALY_ENV=isolated
  ISOLATION_MODE=true
  DATABASE_URL=postgresql://...
  REDIS_URL=redis://...

🆘 Need Help?
===========
  - Type 'exit' to leave the isolation environment
  - Run individual scripts for specific tasks
  - Check logs in /workspace/.isolation/logs/
  - Report issues to the development team

🔗 External Access:
=================
  - API Server: http://localhost:8000 (if running)
  - Documentation: http://localhost:8001 (if running)
  - Database: localhost:5432 (if running)
  - Redis: localhost:6379 (if running)

EOF

# Show current directory contents
echo ""
echo "📂 Current Directory Contents:"
echo "=============================="
ls -la /workspace/ 2>/dev/null || echo "Workspace not accessible"

echo ""
echo "🔍 Quick Health Check:"
echo "====================="
echo "Python available: $(command -v python >/dev/null && echo "✅ Yes" || echo "❌ No")"
echo "Git available: $(command -v git >/dev/null && echo "✅ Yes" || echo "❌ No")"
echo "Docker available: $(command -v docker >/dev/null && echo "✅ Yes" || echo "❌ No")"
echo "Database reachable: $(nc -z postgres-isolated 5432 2>/dev/null && echo "✅ Yes" || echo "❌ No")"
echo "Redis reachable: $(nc -z redis-isolated 6379 2>/dev/null && echo "✅ Yes" || echo "❌ No")"
