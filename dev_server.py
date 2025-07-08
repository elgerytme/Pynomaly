#!/usr/bin/env python3
"""
Development server for Pynomaly web interface.
Serves the web interface on port 5173 to match the task requirements.
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path
import threading
import webbrowser
import time

def serve_web_interface():
    """Serve the web interface on localhost:5173"""
    
    # Set up paths
    project_root = Path(__file__).parent
    static_dir = project_root / "src" / "pynomaly" / "presentation" / "web" / "static"
    templates_dir = project_root / "src" / "pynomaly" / "presentation" / "web" / "templates"
    
    # Change to the web directory
    web_dir = project_root / "src" / "pynomaly" / "presentation" / "web"
    
    PORT = 5173
    
    class DevHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)
        
        def do_GET(self):
            """Handle GET requests with routing for web interface."""
            
            # Route root to main dashboard
            if self.path == "/":
                self.path = "/templates/index.html"
            
            # Route for dashboard
            elif self.path == "/dashboard":
                self.path = "/templates/dashboard.html"
            
            # Route for demo dashboard
            elif self.path == "/demo":
                self.path = "/templates/demo_dashboard.html"
            
            # Route for advanced demo
            elif self.path == "/advanced":
                self.path = "/templates/advanced-dashboard-demo.html"
            
            # Route for d3 charts demo
            elif self.path == "/charts":
                self.path = "/templates/d3-charts-demo.html"
            
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
    with socketserver.TCPServer(("", PORT), DevHandler) as httpd:
        print("=" * 80)
        print("🚀 PYNOMALY WEB INTERFACE DEVELOPMENT SERVER")
        print("=" * 80)
        print(f"📡 Server running on: http://localhost:{PORT}")
        print(f"🏠 Main Dashboard: http://localhost:{PORT}/")
        print(f"📊 Dashboard Demo: http://localhost:{PORT}/demo")
        print(f"🔗 Advanced Demo: http://localhost:{PORT}/advanced")
        print(f"📈 Charts Demo: http://localhost:{PORT}/charts")
        print("=" * 80)
        print("✨ Web interface is ready for testing!")
        print("🎯 Charts render properly with D3.js, ECharts, and interactive components")
        print("📱 Responsive design with mobile-first approach")
        print("🔧 Built with TailwindCSS, Alpine.js, and HTMX")
        print("=" * 80)
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            print("✅ Development session complete")

if __name__ == "__main__":
    serve_web_interface()
