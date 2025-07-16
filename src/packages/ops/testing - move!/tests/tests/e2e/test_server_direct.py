#!/usr/bin/env python3
import sys

sys.path.insert(0, "src")

# Test direct server startup
try:
    import uvicorn

    from monorepo.presentation.api import create_app

    print("✅ Imports successful")

    # Create app
    app = create_app()
    print("✅ App created successfully")

    # Start server
    print("🚀 Starting server on port 8005...")
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="info")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
