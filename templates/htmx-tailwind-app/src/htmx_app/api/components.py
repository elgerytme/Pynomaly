"""Component endpoints for HTMX partial loading."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates

router = APIRouter()


def get_templates(request: Request) -> Jinja2Templates:
    """Get templates instance from app state."""
    return request.app.state.templates


@router.get("/demo-section")
async def demo_section(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Load interactive demo section."""
    return templates.TemplateResponse(
        "components/demo_section.html",
        {"request": request}
    )


@router.get("/user-card/{user_id}")
async def user_card(
    user_id: int, 
    request: Request, 
    templates: Jinja2Templates = Depends(get_templates)
):
    """Load user card component."""
    # Simulate fetching user data
    user_data = {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "avatar_url": f"https://ui-avatars.com/api/?name=User+{user_id}&background=random",
        "bio": f"Software developer #{user_id}",
        "followers": user_id * 10,
        "following": user_id * 5
    }
    
    return templates.TemplateResponse(
        "components/user_card.html",
        {"request": request, "user": user_data}
    )


@router.get("/notification")
async def notification(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Load notification component."""
    notification_data = {
        "id": 1,
        "type": "info",
        "title": "New Feature Available",
        "message": "Check out our new dashboard improvements!",
        "timestamp": datetime.now(),
        "read": False
    }
    
    return templates.TemplateResponse(
        "components/notification.html",
        {"request": request, "notification": notification_data}
    )


@router.get("/loading")
async def loading(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Loading component with delay."""
    await asyncio.sleep(2)  # Simulate slow loading
    
    return templates.TemplateResponse(
        "components/loading_result.html",
        {"request": request, "data": "Content loaded successfully!"}
    )


@router.get("/stats-widget")
async def stats_widget(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Stats widget component."""
    stats = [
        {"label": "Total Users", "value": "12,345", "change": "+12%", "trend": "up"},
        {"label": "Revenue", "value": "$45,678", "change": "+8%", "trend": "up"},
        {"label": "Orders", "value": "1,234", "change": "-3%", "trend": "down"},
        {"label": "Conversion", "value": "3.45%", "change": "+0.5%", "trend": "up"},
    ]
    
    return templates.TemplateResponse(
        "components/stats_widget.html",
        {"request": request, "stats": stats}
    )


@router.get("/task-list")
async def task_list(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Task list component."""
    tasks = [
        {
            "id": 1,
            "title": "Review pull requests",
            "completed": False,
            "priority": "high",
            "due_date": "2024-01-15"
        },
        {
            "id": 2,
            "title": "Update documentation",
            "completed": True,
            "priority": "medium",
            "due_date": "2024-01-14"
        },
        {
            "id": 3,
            "title": "Deploy to production",
            "completed": False,
            "priority": "high",
            "due_date": "2024-01-16"
        },
        {
            "id": 4,
            "title": "Team meeting",
            "completed": False,
            "priority": "low",
            "due_date": "2024-01-17"
        }
    ]
    
    return templates.TemplateResponse(
        "components/task_list.html",
        {"request": request, "tasks": tasks}
    )


@router.get("/chart-data")
async def chart_data(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Chart component with sample data."""
    chart_data = {
        "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "datasets": [
            {
                "label": "Sales",
                "data": [12, 19, 3, 5, 2, 3],
                "backgroundColor": "rgba(59, 130, 246, 0.5)",
                "borderColor": "rgba(59, 130, 246, 1)"
            },
            {
                "label": "Revenue",
                "data": [8, 15, 7, 12, 9, 14],
                "backgroundColor": "rgba(16, 185, 129, 0.5)",
                "borderColor": "rgba(16, 185, 129, 1)"
            }
        ]
    }
    
    return templates.TemplateResponse(
        "components/chart.html",
        {"request": request, "chart_data": chart_data}
    )


@router.get("/infinite-scroll-items")
async def infinite_scroll_items(
    page: int = 1,
    request: Request = None,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Load items for infinite scroll."""
    items_per_page = 10
    start_id = (page - 1) * items_per_page + 1
    
    items = []
    for i in range(items_per_page):
        item_id = start_id + i
        items.append({
            "id": item_id,
            "title": f"Item {item_id}",
            "description": f"This is the description for item {item_id}",
            "image_url": f"https://picsum.photos/300/200?random={item_id}",
            "created_at": datetime.now()
        })
    
    return templates.TemplateResponse(
        "components/infinite_scroll_items.html",
        {
            "request": request,
            "items": items,
            "next_page": page + 1 if page < 10 else None  # Limit to 10 pages for demo
        }
    )