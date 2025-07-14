"""Page endpoints for main application views."""

from __future__ import annotations

from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates

from htmx_app.core.config import get_settings

router = APIRouter()


def get_templates(request: Request) -> Jinja2Templates:
    """Get templates instance from app state."""
    return request.app.state.templates


@router.get("/")
async def home(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Home page."""
    return templates.TemplateResponse(
        "pages/home.html",
        {"request": request, "title": "Home"}
    )


@router.get("/dashboard")
async def dashboard(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Dashboard page."""
    return templates.TemplateResponse(
        "pages/dashboard.html",
        {
            "request": request,
            "title": "Dashboard",
            "user": {"name": "John Doe", "email": "john@example.com"}
        }
    )


@router.get("/examples")
async def examples(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Examples page."""
    examples_data = [
        {
            "title": "Dynamic Forms",
            "description": "Form validation with real-time feedback",
            "url": "/examples/forms"
        },
        {
            "title": "Infinite Scroll",
            "description": "Load content as user scrolls",
            "url": "/examples/infinite-scroll"
        },
        {
            "title": "Live Search",
            "description": "Search with instant results",
            "url": "/examples/search"
        },
        {
            "title": "Modal Dialogs",
            "description": "Dynamic modal content loading",
            "url": "/examples/modals"
        }
    ]
    
    return templates.TemplateResponse(
        "pages/examples.html",
        {
            "request": request,
            "title": "Examples",
            "examples": examples_data
        }
    )


@router.get("/profile")
async def profile(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """User profile page."""
    user_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "bio": "Software developer passionate about web technologies",
        "location": "San Francisco, CA",
        "joined": "January 2024",
        "avatar_url": "https://ui-avatars.com/api/?name=John+Doe&background=3b82f6&color=fff"
    }
    
    return templates.TemplateResponse(
        "pages/profile.html",
        {
            "request": request,
            "title": "Profile",
            "user": user_data
        }
    )


@router.get("/settings")
async def settings(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Settings page."""
    settings_data = {
        "notifications": {
            "email": True,
            "push": False,
            "marketing": True
        },
        "privacy": {
            "profile_public": False,
            "show_email": False
        },
        "theme": "light"
    }
    
    return templates.TemplateResponse(
        "pages/settings.html",
        {
            "request": request,
            "title": "Settings",
            "settings": settings_data
        }
    )