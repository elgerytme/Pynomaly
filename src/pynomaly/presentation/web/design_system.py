"""
Pynomaly Design System
Production-ready UI component specifications and standards
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ColorPalette(Enum):
    """Design system color palette with accessibility-compliant colors"""
    
    # Primary colors
    PRIMARY_50 = "#f0f9ff"
    PRIMARY_100 = "#e0f2fe"
    PRIMARY_500 = "#0ea5e9"
    PRIMARY_600 = "#0284c7"
    PRIMARY_700 = "#0369a1"
    PRIMARY_900 = "#0c4a6e"
    
    # Secondary colors
    SECONDARY_50 = "#fafaf9"
    SECONDARY_100 = "#f5f5f4"
    SECONDARY_500 = "#78716c"
    SECONDARY_600 = "#57534e"
    SECONDARY_700 = "#44403c"
    SECONDARY_900 = "#1c1917"
    
    # Success colors
    SUCCESS_50 = "#f0fdf4"
    SUCCESS_100 = "#dcfce7"
    SUCCESS_500 = "#22c55e"
    SUCCESS_600 = "#16a34a"
    SUCCESS_700 = "#15803d"
    
    # Warning colors
    WARNING_50 = "#fffbeb"
    WARNING_100 = "#fef3c7"
    WARNING_500 = "#f59e0b"
    WARNING_600 = "#d97706"
    WARNING_700 = "#b45309"
    
    # Error colors
    ERROR_50 = "#fef2f2"
    ERROR_100 = "#fee2e2"
    ERROR_500 = "#ef4444"
    ERROR_600 = "#dc2626"
    ERROR_700 = "#b91c1c"
    
    # Neutral colors
    NEUTRAL_50 = "#fafafa"
    NEUTRAL_100 = "#f5f5f5"
    NEUTRAL_200 = "#e5e5e5"
    NEUTRAL_300 = "#d4d4d4"
    NEUTRAL_400 = "#a3a3a3"
    NEUTRAL_500 = "#737373"
    NEUTRAL_600 = "#525252"
    NEUTRAL_700 = "#404040"
    NEUTRAL_800 = "#262626"
    NEUTRAL_900 = "#171717"

class Typography(Enum):
    """Typography scale following 8pt grid system"""
    
    # Display text
    DISPLAY_LARGE = "text-6xl font-bold leading-none tracking-tight"  # 60px
    DISPLAY_MEDIUM = "text-5xl font-bold leading-none tracking-tight"  # 48px
    DISPLAY_SMALL = "text-4xl font-bold leading-tight tracking-tight"  # 36px
    
    # Headline text
    HEADLINE_LARGE = "text-3xl font-semibold leading-tight"  # 30px
    HEADLINE_MEDIUM = "text-2xl font-semibold leading-tight"  # 24px
    HEADLINE_SMALL = "text-xl font-semibold leading-tight"  # 20px
    
    # Title text
    TITLE_LARGE = "text-lg font-medium leading-normal"  # 18px
    TITLE_MEDIUM = "text-base font-medium leading-normal"  # 16px
    TITLE_SMALL = "text-sm font-medium leading-normal"  # 14px
    
    # Body text
    BODY_LARGE = "text-base font-normal leading-relaxed"  # 16px
    BODY_MEDIUM = "text-sm font-normal leading-relaxed"  # 14px
    BODY_SMALL = "text-xs font-normal leading-relaxed"  # 12px
    
    # Label text
    LABEL_LARGE = "text-sm font-medium leading-none"  # 14px
    LABEL_MEDIUM = "text-xs font-medium leading-none"  # 12px
    LABEL_SMALL = "text-xs font-medium leading-none uppercase tracking-wide"  # 12px

class Spacing(Enum):
    """Spacing scale following 8pt grid system"""
    
    NONE = "0"
    XS = "0.125rem"  # 2px
    SM = "0.25rem"   # 4px
    MD = "0.5rem"    # 8px
    LG = "1rem"      # 16px
    XL = "1.5rem"    # 24px
    XXL = "2rem"     # 32px
    XXXL = "3rem"    # 48px
    HUGE = "4rem"    # 64px

@dataclass
class ComponentSpec:
    """Specification for a UI component"""
    name: str
    description: str
    variants: List[str]
    props: Dict[str, Any]
    accessibility: Dict[str, str]
    examples: List[Dict[str, str]]

class DesignTokens:
    """Central design tokens for the Pynomaly design system"""
    
    # Color tokens
    COLORS = {
        'primary': ColorPalette.PRIMARY_500.value,
        'primary-hover': ColorPalette.PRIMARY_600.value,
        'primary-active': ColorPalette.PRIMARY_700.value,
        'primary-light': ColorPalette.PRIMARY_100.value,
        'primary-dark': ColorPalette.PRIMARY_900.value,
        
        'secondary': ColorPalette.SECONDARY_500.value,
        'secondary-hover': ColorPalette.SECONDARY_600.value,
        'secondary-light': ColorPalette.SECONDARY_100.value,
        
        'success': ColorPalette.SUCCESS_500.value,
        'success-light': ColorPalette.SUCCESS_100.value,
        
        'warning': ColorPalette.WARNING_500.value,
        'warning-light': ColorPalette.WARNING_100.value,
        
        'error': ColorPalette.ERROR_500.value,
        'error-light': ColorPalette.ERROR_100.value,
        
        'background': ColorPalette.NEUTRAL_50.value,
        'surface': ColorPalette.NEUTRAL_100.value,
        'text-primary': ColorPalette.NEUTRAL_900.value,
        'text-secondary': ColorPalette.NEUTRAL_600.value,
        'text-muted': ColorPalette.NEUTRAL_400.value,
        'border': ColorPalette.NEUTRAL_200.value,
        'border-hover': ColorPalette.NEUTRAL_300.value,
    }
    
    # Shadow tokens
    SHADOWS = {
        'sm': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        'md': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        'lg': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
        'xl': '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
    }
    
    # Border radius tokens
    RADIUS = {
        'none': '0',
        'sm': '0.125rem',  # 2px
        'md': '0.375rem',  # 6px
        'lg': '0.5rem',    # 8px
        'xl': '0.75rem',   # 12px
        'full': '9999px',
    }
    
    # Animation tokens
    TRANSITIONS = {
        'fast': '150ms cubic-bezier(0.4, 0, 0.2, 1)',
        'normal': '300ms cubic-bezier(0.4, 0, 0.2, 1)',
        'slow': '500ms cubic-bezier(0.4, 0, 0.2, 1)',
    }

class ComponentLibrary:
    """Production-ready component specifications"""
    
    @staticmethod
    def get_button_spec() -> ComponentSpec:
        """Button component specification"""
        return ComponentSpec(
            name="Button",
            description="Interactive button component with multiple variants and states",
            variants=["primary", "secondary", "outline", "ghost", "danger"],
            props={
                "size": ["sm", "md", "lg"],
                "disabled": bool,
                "loading": bool,
                "icon": Optional[str],
                "full_width": bool,
                "type": ["button", "submit", "reset"]
            },
            accessibility={
                "role": "button",
                "aria-label": "Required for icon-only buttons",
                "aria-describedby": "Optional description reference",
                "aria-pressed": "For toggle buttons",
                "tabindex": "0 for interactive buttons",
                "focus-visible": "Clear focus indication required"
            },
            examples=[
                {
                    "name": "Primary Button",
                    "html": '<button class="btn btn-primary btn-md" type="button">Save Changes</button>'
                },
                {
                    "name": "Loading Button", 
                    "html": '<button class="btn btn-primary btn-md" disabled aria-label="Saving..."><span class="spinner"></span> Saving...</button>'
                }
            ]
        )
    
    @staticmethod
    def get_input_spec() -> ComponentSpec:
        """Input component specification"""
        return ComponentSpec(
            name="Input",
            description="Form input component with validation states and accessibility features",
            variants=["text", "email", "password", "number", "search", "textarea"],
            props={
                "size": ["sm", "md", "lg"],
                "disabled": bool,
                "readonly": bool,
                "required": bool,
                "error": bool,
                "success": bool,
                "placeholder": str,
                "help_text": Optional[str],
                "prefix_icon": Optional[str],
                "suffix_icon": Optional[str]
            },
            accessibility={
                "aria-label": "Required if no associated label",
                "aria-labelledby": "Reference to label element",
                "aria-describedby": "Reference to help text or error",
                "aria-invalid": "true for error state",
                "aria-required": "true for required fields",
                "autocomplete": "Appropriate autocomplete value"
            },
            examples=[
                {
                    "name": "Text Input with Label",
                    "html": '<div class="input-group"><label for="email" class="input-label">Email Address</label><input type="email" id="email" class="input input-md" placeholder="Enter your email" /></div>'
                },
                {
                    "name": "Input with Error",
                    "html": '<div class="input-group"><input type="email" class="input input-md input-error" aria-invalid="true" aria-describedby="email-error" /><div id="email-error" class="input-error-text">Please enter a valid email address</div></div>'
                }
            ]
        )
    
    @staticmethod
    def get_card_spec() -> ComponentSpec:
        """Card component specification"""
        return ComponentSpec(
            name="Card",
            description="Flexible container component for grouping related content",
            variants=["default", "elevated", "outlined", "interactive"],
            props={
                "padding": ["none", "sm", "md", "lg"],
                "clickable": bool,
                "header": bool,
                "footer": bool,
                "image": bool,
                "loading": bool
            },
            accessibility={
                "role": "article or region for semantic cards",
                "aria-label": "Card title or description",
                "tabindex": "0 for interactive cards",
                "aria-pressed": "For selectable cards"
            },
            examples=[
                {
                    "name": "Basic Card",
                    "html": '<div class="card card-default card-md"><div class="card-header"><h3 class="card-title">Dataset Analysis</h3></div><div class="card-content">View and analyze your uploaded datasets.</div></div>'
                },
                {
                    "name": "Interactive Card",
                    "html": '<div class="card card-interactive card-md" tabindex="0" role="button" aria-label="Select dataset for analysis"><div class="card-content">Customer_Data.csv<br><span class="text-muted">1,247 rows • 12 columns</span></div></div>'
                }
            ]
        )
    
    @staticmethod
    def get_modal_spec() -> ComponentSpec:
        """Modal component specification"""
        return ComponentSpec(
            name="Modal",
            description="Overlay dialog component for focused interactions",
            variants=["default", "fullscreen", "sidebar"],
            props={
                "size": ["sm", "md", "lg", "xl"],
                "dismissible": bool,
                "backdrop_blur": bool,
                "persistent": bool,
                "header": bool,
                "footer": bool
            },
            accessibility={
                "role": "dialog",
                "aria-modal": "true",
                "aria-labelledby": "Modal title element ID",
                "aria-describedby": "Modal content description ID",
                "focus-trap": "Focus must be trapped within modal",
                "escape-key": "ESC key closes dismissible modals"
            },
            examples=[
                {
                    "name": "Confirmation Modal",
                    "html": '<div class="modal-backdrop"><div class="modal modal-md" role="dialog" aria-modal="true" aria-labelledby="modal-title"><div class="modal-header"><h2 id="modal-title" class="modal-title">Confirm Deletion</h2><button class="modal-close" aria-label="Close modal">&times;</button></div><div class="modal-content">Are you sure you want to delete this dataset?</div><div class="modal-footer"><button class="btn btn-outline">Cancel</button><button class="btn btn-danger">Delete</button></div></div></div>'
                }
            ]
        )
    
    @staticmethod
    def get_navigation_spec() -> ComponentSpec:
        """Navigation component specification"""
        return ComponentSpec(
            name="Navigation",
            description="Primary navigation component with responsive behavior",
            variants=["horizontal", "vertical", "mobile"],
            props={
                "collapsed": bool,
                "sticky": bool,
                "breadcrumbs": bool,
                "search": bool,
                "user_menu": bool,
                "notifications": bool
            },
            accessibility={
                "role": "navigation",
                "aria-label": "Main navigation or section name",
                "aria-current": "page for current page link",
                "skip-link": "Skip to main content link required",
                "keyboard-navigation": "Arrow keys and tab navigation support"
            },
            examples=[
                {
                    "name": "Main Navigation",
                    "html": '<nav class="navbar" role="navigation" aria-label="Main navigation"><div class="navbar-brand"><img src="/logo.svg" alt="Pynomaly" class="navbar-logo" /></div><ul class="navbar-nav"><li class="nav-item"><a href="/dashboard" class="nav-link" aria-current="page">Dashboard</a></li><li class="nav-item"><a href="/datasets" class="nav-link">Datasets</a></li></ul></nav>'
                }
            ]
        )

class AccessibilityGuidelines:
    """WCAG 2.1 AA compliance guidelines for components"""
    
    CONTRAST_RATIOS = {
        "normal_text": 4.5,  # AA standard
        "large_text": 3.0,   # AA standard for 18pt+ or 14pt+ bold
        "ui_components": 3.0, # AA standard for UI components
        "enhanced": 7.0       # AAA standard
    }
    
    TOUCH_TARGETS = {
        "minimum": "44px",    # WCAG 2.1 AA minimum
        "recommended": "48px", # Recommended for better usability
        "spacing": "8px"      # Minimum spacing between targets
    }
    
    FOCUS_MANAGEMENT = {
        "visible_indicator": "2px solid outline with high contrast",
        "skip_links": "Required for keyboard navigation",
        "focus_trap": "Required for modals and overlays",
        "logical_order": "Tab order follows visual order"
    }
    
    ANIMATION_PREFERENCES = {
        "respect_reduced_motion": True,
        "max_duration": "500ms for UI animations",
        "provide_controls": "For auto-playing content"
    }

class PWAFeatures:
    """Progressive Web App feature specifications"""
    
    MANIFEST_CONFIG = {
        "name": "Pynomaly - Anomaly Detection Platform",
        "short_name": "Pynomaly",
        "description": "Production-ready anomaly detection and analysis platform",
        "start_url": "/",
        "display": "standalone",
        "background_color": ColorPalette.NEUTRAL_50.value,
        "theme_color": ColorPalette.PRIMARY_500.value,
        "orientation": "portrait-primary",
        "categories": ["productivity", "business", "education"],
        "screenshots": [
            {
                "src": "/screenshots/dashboard-wide.png",
                "sizes": "1280x720",
                "type": "image/png",
                "form_factor": "wide",
                "label": "Dashboard view"
            },
            {
                "src": "/screenshots/dashboard-narrow.png", 
                "sizes": "375x667",
                "type": "image/png",
                "form_factor": "narrow",
                "label": "Mobile dashboard view"
            }
        ],
        "icons": [
            {
                "src": "/icons/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "/icons/icon-512.png",
                "sizes": "512x512", 
                "type": "image/png",
                "purpose": "any maskable"
            }
        ]
    }
    
    SERVICE_WORKER_STRATEGIES = {
        "cache_first": ["images", "fonts", "static_assets"],
        "network_first": ["api_calls", "dynamic_content"],
        "stale_while_revalidate": ["html_pages", "stylesheets", "scripts"],
        "offline_fallback": ["main_pages", "error_pages"]
    }
    
    OFFLINE_CAPABILITIES = {
        "view_cached_data": "Previously loaded datasets and results",
        "basic_navigation": "Core app structure and static pages",
        "offline_indicator": "Clear indication when offline",
        "sync_when_online": "Automatic sync when connection restored"
    }

def generate_tailwind_config() -> Dict[str, Any]:
    """Generate Tailwind CSS configuration for the design system"""
    return {
        "content": [
            "./src/pynomaly/presentation/web/**/*.{html,js,py}",
            "./tests/ui/**/*.{html,js,py}"
        ],
        "theme": {
            "extend": {
                "colors": {
                    "primary": {
                        "50": ColorPalette.PRIMARY_50.value,
                        "100": ColorPalette.PRIMARY_100.value,
                        "500": ColorPalette.PRIMARY_500.value,
                        "600": ColorPalette.PRIMARY_600.value,
                        "700": ColorPalette.PRIMARY_700.value,
                        "900": ColorPalette.PRIMARY_900.value,
                    },
                    "secondary": {
                        "50": ColorPalette.SECONDARY_50.value,
                        "100": ColorPalette.SECONDARY_100.value,
                        "500": ColorPalette.SECONDARY_500.value,
                        "600": ColorPalette.SECONDARY_600.value,
                        "700": ColorPalette.SECONDARY_700.value,
                        "900": ColorPalette.SECONDARY_900.value,
                    },
                    "success": {
                        "50": ColorPalette.SUCCESS_50.value,
                        "100": ColorPalette.SUCCESS_100.value,
                        "500": ColorPalette.SUCCESS_500.value,
                        "600": ColorPalette.SUCCESS_600.value,
                        "700": ColorPalette.SUCCESS_700.value,
                    },
                    "warning": {
                        "50": ColorPalette.WARNING_50.value,
                        "100": ColorPalette.WARNING_100.value,
                        "500": ColorPalette.WARNING_500.value,
                        "600": ColorPalette.WARNING_600.value,
                        "700": ColorPalette.WARNING_700.value,
                    },
                    "error": {
                        "50": ColorPalette.ERROR_50.value,
                        "100": ColorPalette.ERROR_100.value,
                        "500": ColorPalette.ERROR_500.value,
                        "600": ColorPalette.ERROR_600.value,
                        "700": ColorPalette.ERROR_700.value,
                    }
                },
                "fontFamily": {
                    "sans": ["Inter", "system-ui", "sans-serif"],
                    "mono": ["JetBrains Mono", "Menlo", "monospace"]
                },
                "boxShadow": DesignTokens.SHADOWS,
                "borderRadius": DesignTokens.RADIUS,
                "transitionDuration": {
                    "fast": "150ms",
                    "normal": "300ms", 
                    "slow": "500ms"
                }
            }
        },
        "plugins": [
            "@tailwindcss/forms",
            "@tailwindcss/typography",
            "@tailwindcss/aspect-ratio"
        ]
    }