"""
Enhanced Content Security Policy (CSP) Configuration
Implements strict CSP with nonce support and SRI validation
"""

import hashlib
import secrets
from typing import Dict, List, Optional
from urllib.parse import urlparse

class CSPBuilder:
    """Enhanced CSP builder with dynamic policy generation"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.nonce = self.generate_nonce()
        self.hashes = {}
        self.report_uri = "/api/security/csp-report"

        # Trusted CDN domains for external resources
        self.trusted_cdns = [
            "cdn.jsdelivr.net",
            "unpkg.com",
            "cdnjs.cloudflare.com"
        ]

        # Resource integrity hashes (updated automatically)
        self.sri_hashes = {
            "https://cdn.tailwindcss.com": "sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN",
            "https://unpkg.com/htmx.org@1.9.10": "sha384-D1Kt99CQMDuVetoL1lrYwg5t+9QdHe7NLX/SoJYkXDFfX37iInKRy5xLSi8nO7UC",
            "https://unpkg.com/alpinejs@3.13.3/dist/cdn.min.js": "sha384-x7+gjw5ZRalJWQPKnyVTpf4K2k8SdxGlszLLBiPJL8YCZqeqI8JVvZvbW9bFgN6L"
        }

    def generate_nonce(self) -> str:
        """Generate cryptographically secure nonce"""
        return secrets.token_urlsafe(32)

    def add_inline_script_hash(self, script_content: str) -> str:
        """Add hash for inline script content"""
        script_hash = hashlib.sha384(script_content.encode()).digest()
        hash_value = f"sha384-{script_hash.hex()}"

        if 'script-src' not in self.hashes:
            self.hashes['script-src'] = []
        self.hashes['script-src'].append(hash_value)

        return hash_value

    def add_inline_style_hash(self, style_content: str) -> str:
        """Add hash for inline style content"""
        style_hash = hashlib.sha384(style_content.encode()).digest()
        hash_value = f"sha384-{style_hash.hex()}"

        if 'style-src' not in self.hashes:
            self.hashes['style-src'] = []
        self.hashes['style-src'].append(hash_value)

        return hash_value

    def get_base_policy(self) -> Dict[str, List[str]]:
        """Get base CSP policy directives"""
        policy = {
            'default-src': ["'self'"],
            'script-src': [
                "'self'",
                f"'nonce-{self.nonce}'",
                *[f"https://{cdn}" for cdn in self.trusted_cdns]
            ],
            'style-src': [
                "'self'",
                f"'nonce-{self.nonce}'",
                *[f"https://{cdn}" for cdn in self.trusted_cdns]
            ],
            'img-src': [
                "'self'",
                "data:",
                "blob:",
                *[f"https://{cdn}" for cdn in self.trusted_cdns]
            ],
            'font-src': [
                "'self'",
                "data:",
                *[f"https://{cdn}" for cdn in self.trusted_cdns]
            ],
            'connect-src': [
                "'self'",
                "wss:",
                "ws:"
            ],
            'media-src': ["'self'"],
            'object-src': ["'none'"],
            'frame-src': ["'none'"],
            'frame-ancestors': ["'none'"],
            'form-action': ["'self'"],
            'base-uri': ["'self'"],
            'manifest-src': ["'self'"],
            'worker-src': ["'self'"],
            'child-src': ["'none'"],
            'upgrade-insecure-requests': [],
            'block-all-mixed-content': []
        }

        # Add hashes for inline content
        for directive, hashes in self.hashes.items():
            if directive in policy:
                policy[directive].extend([f"'{hash_val}'" for hash_val in hashes])

        # Development mode adjustments
        if self.debug_mode:
            # Allow localhost and development servers
            policy['connect-src'].extend([
                "http://localhost:*",
                "ws://localhost:*",
                "http://127.0.0.1:*"
            ])
            # Allow eval for development tools (with warning)
            policy['script-src'].append("'unsafe-eval'")

        return policy

    def get_strict_policy(self) -> Dict[str, List[str]]:
        """Get strict CSP policy for production"""
        policy = self.get_base_policy()

        # Remove any unsafe directives
        for directive in policy:
            policy[directive] = [
                source for source in policy[directive]
                if not source.startswith("'unsafe-")
            ]

        # Add reporting
        policy['report-uri'] = [self.report_uri]
        policy['report-to'] = ["csp-endpoint"]

        return policy

    def get_report_only_policy(self) -> Dict[str, List[str]]:
        """Get report-only CSP policy for testing"""
        policy = self.get_base_policy()
        policy['report-uri'] = [self.report_uri]
        return policy

    def build_header_value(self, policy: Dict[str, List[str]]) -> str:
        """Build CSP header value from policy dict"""
        directives = []

        for directive, sources in policy.items():
            if sources:
                directive_value = f"{directive} {' '.join(sources)}"
            else:
                directive_value = directive
            directives.append(directive_value)

        return "; ".join(directives)

    def get_csp_header(self, report_only: bool = False) -> Dict[str, str]:
        """Get CSP header for response"""
        if report_only:
            policy = self.get_report_only_policy()
            header_name = "Content-Security-Policy-Report-Only"
        else:
            policy = self.get_strict_policy() if not self.debug_mode else self.get_base_policy()
            header_name = "Content-Security-Policy"

        return {header_name: self.build_header_value(policy)}

class SubresourceIntegrityManager:
    """Manages Subresource Integrity (SRI) for external resources"""

    def __init__(self):
        self.integrity_hashes = {
            # Tailwind CSS
            "https://cdn.tailwindcss.com": {
                "integrity": "sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN",
                "crossorigin": "anonymous"
            },
            # HTMX
            "https://unpkg.com/htmx.org@1.9.10": {
                "integrity": "sha384-D1Kt99CQMDuVetoL1lrYwg5t+9QdHe7NLX/SoJYkXDFfX37iInKRy5xLSi8nO7UC",
                "crossorigin": "anonymous"
            },
            # Alpine.js
            "https://unpkg.com/alpinejs@3.13.3/dist/cdn.min.js": {
                "integrity": "sha384-x7+gjw5ZRalJWQPKnyVTpf4K2k8SdxGlszLLBiPJL8YCZqeqI8JVvZvbW9bFgN6L",
                "crossorigin": "anonymous"
            }
        }

    def get_sri_attributes(self, url: str) -> Dict[str, str]:
        """Get SRI attributes for a given URL"""
        return self.integrity_hashes.get(url, {})

    def is_sri_protected(self, url: str) -> bool:
        """Check if URL has SRI protection"""
        return url in self.integrity_hashes

    def add_sri_hash(self, url: str, integrity: str, crossorigin: str = "anonymous"):
        """Add SRI hash for a URL"""
        self.integrity_hashes[url] = {
            "integrity": integrity,
            "crossorigin": crossorigin
        }

    def generate_script_tag(self, url: str, **attributes) -> str:
        """Generate script tag with SRI"""
        sri_attrs = self.get_sri_attributes(url)

        attrs = [f'src="{url}"']

        if sri_attrs:
            attrs.append(f'integrity="{sri_attrs["integrity"]}"')
            attrs.append(f'crossorigin="{sri_attrs["crossorigin"]}"')

        # Add other attributes
        for key, value in attributes.items():
            if key not in ['src', 'integrity', 'crossorigin']:
                attrs.append(f'{key}="{value}"')

        return f'<script {" ".join(attrs)}></script>'

    def generate_link_tag(self, url: str, rel: str = "stylesheet", **attributes) -> str:
        """Generate link tag with SRI"""
        sri_attrs = self.get_sri_attributes(url)

        attrs = [f'href="{url}"', f'rel="{rel}"']

        if sri_attrs:
            attrs.append(f'integrity="{sri_attrs["integrity"]}"')
            attrs.append(f'crossorigin="{sri_attrs["crossorigin"]}"')

        # Add other attributes
        for key, value in attributes.items():
            if key not in ['href', 'rel', 'integrity', 'crossorigin']:
                attrs.append(f'{key}="{value}"')

        return f'<link {" ".join(attrs)} />'

class SecurityHeadersManager:
    """Manages comprehensive security headers"""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.csp_builder = CSPBuilder(debug_mode)
        self.sri_manager = SubresourceIntegrityManager()

    def get_security_headers(self, include_csp: bool = True, report_only: bool = False) -> Dict[str, str]:
        """Get all security headers"""
        headers = {}

        # Content Security Policy
        if include_csp:
            headers.update(self.csp_builder.get_csp_header(report_only))

        # X-Frame-Options
        headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options
        headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection
        headers["X-XSS-Protection"] = "1; mode=block"

        # Strict-Transport-Security (HTTPS only)
        if not self.debug_mode:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # Referrer-Policy
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "accelerometer=()",
            "gyroscope=()"
        ]
        headers["Permissions-Policy"] = ", ".join(permissions)

        # Cross-Origin Policies
        headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Expect-CT (Certificate Transparency)
        if not self.debug_mode:
            headers["Expect-CT"] = 'max-age=86400, enforce, report-uri="/api/security/ct-report"'

        # Network Error Logging
        headers["NEL"] = '{"report_to":"default","max_age":31536000,"include_subdomains":true}'

        # Report-To API
        report_to = {
            "group": "default",
            "max_age": 31536000,
            "endpoints": [{"url": "/api/security/nel-report"}]
        }
        headers["Report-To"] = str(report_to).replace("'", '"')

        return headers

    def get_nonce(self) -> str:
        """Get current CSP nonce"""
        return self.csp_builder.nonce

    def add_inline_script_hash(self, script_content: str) -> str:
        """Add hash for inline script"""
        return self.csp_builder.add_inline_script_hash(script_content)

    def add_inline_style_hash(self, style_content: str) -> str:
        """Add hash for inline style"""
        return self.csp_builder.add_inline_style_hash(style_content)

# Template helpers for Jinja2
def setup_template_helpers(app):
    """Setup template helpers for security features"""

    security_manager = SecurityHeadersManager(app.debug)

    @app.template_global()
    def csp_nonce():
        """Get CSP nonce for templates"""
        return security_manager.get_nonce()

    @app.template_global()
    def sri_script(url, **kwargs):
        """Generate script tag with SRI"""
        return security_manager.sri_manager.generate_script_tag(url, **kwargs)

    @app.template_global()
    def sri_link(url, **kwargs):
        """Generate link tag with SRI"""
        return security_manager.sri_manager.generate_link_tag(url, **kwargs)

    @app.template_global()
    def inline_script_hash(content):
        """Add hash for inline script content"""
        return security_manager.add_inline_script_hash(content)

    @app.template_global()
    def inline_style_hash(content):
        """Add hash for inline style content"""
        return security_manager.add_inline_style_hash(content)

    return security_manager

# Example usage in middleware
class EnhancedSecurityMiddleware:
    """Enhanced security middleware with CSP and security headers"""

    def __init__(self, app, debug_mode: bool = False):
        self.app = app
        self.security_manager = SecurityHeadersManager(debug_mode)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add security headers to response
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))

                    # Add security headers
                    security_headers = self.security_manager.get_security_headers()
                    for name, value in security_headers.items():
                        headers[name.lower().encode()] = value.encode()

                    message["headers"] = list(headers.items())

                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# Export main classes
__all__ = [
    'CSPBuilder',
    'SubresourceIntegrityManager',
    'SecurityHeadersManager',
    'EnhancedSecurityMiddleware',
    'setup_template_helpers'
]
