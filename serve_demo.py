#!/usr/bin/env python3
"""
Simple HTTP server to serve the demo dashboard for P-001 verification.
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path
import threading
import webbrowser
import time

def serve_demo():
    """Serve the demo dashboard on localhost:8001"""
    
    # Set up paths
    project_root = Path(__file__).parent
    static_dir = project_root / "src" / "pynomaly" / "presentation" / "web" / "static"
    templates_dir = project_root / "src" / "pynomaly" / "presentation" / "web" / "templates"
    
    # Change to the web directory
    web_dir = project_root / "src" / "pynomaly" / "presentation" / "web"
    os.chdir(web_dir)
    
    PORT = 8001
    
    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)
        
        def do_GET(self):
            """Handle GET requests with special routing for demo."""
            
            # Route root to demo dashboard
            if self.path == "/" or self.path == "/demo":
                self.path = "/templates/demo_dashboard.html"
            
            # Route for advanced demo
            elif self.path == "/advanced":
                self.path = "/templates/advanced-dashboard-demo.html"
            
            # Route for dashboard demo
            elif self.path == "/dashboard":
                self.path = "/templates/dashboard-demo.html"
            
            # For static files, serve from static directory
            elif self.path.startswith("/static/"):
                # Remove /static/ prefix and serve from static directory
                file_path = self.path[8:]  # Remove '/static/'
                full_path = static_dir / file_path
                if full_path.exists():
                    self.path = f"/static/{file_path}"
                else:
                    self.send_error(404, f"File not found: {file_path}")
                    return
            
            # Call parent handler
            super().do_GET()
        
        def log_message(self, format, *args):
            """Custom log message format."""
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")
    
    # Start server
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        print("=" * 80)
        print("🚀 PYNOMALY DEMO DASHBOARD SERVER STARTING")
        print("=" * 80)
        print(f"📡 Server running on: http://localhost:{PORT}")
        print(f"🎯 Demo Dashboard: http://localhost:{PORT}/demo")
        print(f"🔗 Advanced Demo: http://localhost:{PORT}/advanced")
        print(f"📊 Dashboard Demo: http://localhost:{PORT}/dashboard")
        print("=" * 80)
        print("✨ Ready for P-001 verification and recording!")
        print("🔥 Dashboard features real-time data, interactive charts, and anomaly detection")
        print("⚡ All demo data is generated dynamically in-browser")
        print("=" * 80)
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        # Open browser automatically
        def open_browser():
            time.sleep(1)
            webbrowser.open(f"http://localhost:{PORT}/demo")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            print("✅ Demo session complete")

if __name__ == "__main__":
    serve_demo()
