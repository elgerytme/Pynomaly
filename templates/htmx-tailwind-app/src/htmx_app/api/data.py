"""Data endpoints for HTMX requests."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Request, Query, Depends
from fastapi.templating import Jinja2Templates

router = APIRouter()


def get_templates(request: Request) -> Jinja2Templates:
    """Get templates instance from app state."""
    return request.app.state.templates


@router.get("/status")
async def get_status(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Get system status."""
    # Simulate varying status
    statuses = ["operational", "degraded", "maintenance", "outage"]
    weights = [0.8, 0.15, 0.04, 0.01]  # Mostly operational
    
    status = random.choices(statuses, weights=weights)[0]
    
    status_data = {
        "status": status,
        "uptime": "99.9%",
        "response_time": f"{random.randint(50, 200)}ms",
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "incidents": random.randint(0, 3)
    }
    
    return templates.TemplateResponse(
        "components/status_indicator.html",
        {"request": request, "status": status_data}
    )


@router.get("/notifications")
async def get_notifications(
    unread_only: bool = Query(False),
    request: Request = None,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Get user notifications."""
    # Mock notifications data
    all_notifications = [
        {
            "id": 1,
            "type": "info",
            "title": "Welcome to HTMX App",
            "message": "Thanks for trying our template!",
            "timestamp": datetime.now() - timedelta(minutes=5),
            "read": False
        },
        {
            "id": 2,
            "type": "success",
            "title": "Profile Updated",
            "message": "Your profile has been successfully updated.",
            "timestamp": datetime.now() - timedelta(hours=1),
            "read": True
        },
        {
            "id": 3,
            "type": "warning",
            "title": "Password Expiring",
            "message": "Your password will expire in 7 days.",
            "timestamp": datetime.now() - timedelta(hours=2),
            "read": False
        },
        {
            "id": 4,
            "type": "error",
            "title": "Failed Login Attempt",
            "message": "Someone tried to access your account.",
            "timestamp": datetime.now() - timedelta(days=1),
            "read": True
        }
    ]
    
    notifications = all_notifications
    if unread_only:
        notifications = [n for n in notifications if not n["read"]]
    
    return templates.TemplateResponse(
        "components/notification_list.html",
        {
            "request": request,
            "notifications": notifications,
            "unread_count": len([n for n in all_notifications if not n["read"]])
        }
    )


@router.get("/analytics")
async def get_analytics(
    period: str = Query("7d"),
    request: Request = None,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Get analytics data."""
    # Generate mock analytics data based on period
    periods = {"24h": 24, "7d": 7, "30d": 30, "90d": 90}
    days = periods.get(period, 7)
    
    # Generate sample data
    analytics_data = {
        "period": period,
        "total_users": random.randint(1000, 5000),
        "active_users": random.randint(500, 2000),
        "page_views": random.randint(10000, 50000),
        "bounce_rate": f"{random.randint(20, 60)}%",
        "avg_session": f"{random.randint(2, 8)}m {random.randint(10, 59)}s",
        "conversion_rate": f"{random.uniform(1.5, 4.5):.1f}%",
        "chart_data": []
    }
    
    # Generate chart data points
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i-1)
        analytics_data["chart_data"].append({
            "date": date.strftime("%Y-%m-%d"),
            "users": random.randint(50, 200),
            "sessions": random.randint(80, 300),
            "pageviews": random.randint(200, 800)
        })
    
    return templates.TemplateResponse(
        "components/analytics_dashboard.html",
        {"request": request, "analytics": analytics_data}
    )


@router.get("/users")
async def get_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    request: Request = None,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Get paginated users list."""
    # Mock users data
    all_users = []
    for i in range(1, 101):  # 100 mock users
        all_users.append({
            "id": i,
            "name": f"User {i}",
            "email": f"user{i}@example.com",
            "role": random.choice(["admin", "user", "moderator"]),
            "status": random.choice(["active", "inactive", "pending"]),
            "last_login": datetime.now() - timedelta(days=random.randint(0, 30)),
            "avatar_url": f"https://ui-avatars.com/api/?name=User+{i}&background=random"
        })
    
    # Filter by search if provided
    if search:
        search_lower = search.lower()
        all_users = [
            user for user in all_users
            if search_lower in user["name"].lower() or search_lower in user["email"].lower()
        ]
    
    # Paginate results
    total_users = len(all_users)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    users = all_users[start_idx:end_idx]
    
    # Calculate pagination info
    total_pages = (total_users + per_page - 1) // per_page
    has_next = page < total_pages
    has_prev = page > 1
    
    return templates.TemplateResponse(
        "components/users_table.html",
        {
            "request": request,
            "users": users,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
                "next_page": page + 1 if has_next else None,
                "prev_page": page - 1 if has_prev else None,
                "total_users": total_users,
                "start_idx": start_idx + 1,
                "end_idx": min(end_idx, total_users)
            },
            "search": search or ""
        }
    )


@router.get("/weather")
async def get_weather(
    city: str = Query("San Francisco"),
    request: Request = None,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Get weather data for a city."""
    # Simulate API call delay
    await asyncio.sleep(1)
    
    # Mock weather data
    conditions = ["sunny", "cloudy", "rainy", "snowy", "partly-cloudy"]
    condition = random.choice(conditions)
    
    weather_data = {
        "city": city,
        "temperature": random.randint(-10, 35),
        "condition": condition,
        "humidity": random.randint(30, 90),
        "wind_speed": random.randint(5, 25),
        "pressure": random.randint(990, 1030),
        "feels_like": random.randint(-15, 40),
        "uv_index": random.randint(1, 11),
        "forecast": []
    }
    
    # Generate 5-day forecast
    for i in range(5):
        date = datetime.now() + timedelta(days=i+1)
        weather_data["forecast"].append({
            "date": date.strftime("%a, %b %d"),
            "high": random.randint(weather_data["temperature"], weather_data["temperature"] + 10),
            "low": random.randint(weather_data["temperature"] - 10, weather_data["temperature"]),
            "condition": random.choice(conditions),
            "precipitation": random.randint(0, 90)
        })
    
    return templates.TemplateResponse(
        "components/weather_widget.html",
        {"request": request, "weather": weather_data}
    )


@router.get("/live-metrics")
async def get_live_metrics(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Get live system metrics."""
    # Simulate real-time metrics
    metrics = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "cpu_usage": random.randint(10, 90),
        "memory_usage": random.randint(30, 85),
        "disk_usage": random.randint(20, 70),
        "network_in": random.randint(100, 1000),
        "network_out": random.randint(50, 800),
        "active_connections": random.randint(10, 200),
        "requests_per_minute": random.randint(50, 500),
        "error_rate": random.uniform(0, 5),
        "response_time": random.randint(50, 300)
    }
    
    return templates.TemplateResponse(
        "components/live_metrics.html",
        {"request": request, "metrics": metrics}
    )


@router.get("/activity-feed")
async def get_activity_feed(
    limit: int = Query(10, ge=1, le=50),
    request: Request = None,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Get recent activity feed."""
    activities = [
        {
            "id": 1,
            "user": "John Doe",
            "action": "created",
            "object": "new project",
            "timestamp": datetime.now() - timedelta(minutes=2),
            "avatar_url": "https://ui-avatars.com/api/?name=John+Doe"
        },
        {
            "id": 2,
            "user": "Jane Smith",
            "action": "updated",
            "object": "user profile",
            "timestamp": datetime.now() - timedelta(minutes=15),
            "avatar_url": "https://ui-avatars.com/api/?name=Jane+Smith"
        },
        {
            "id": 3,
            "user": "Bob Wilson",
            "action": "deleted",
            "object": "old task",
            "timestamp": datetime.now() - timedelta(hours=1),
            "avatar_url": "https://ui-avatars.com/api/?name=Bob+Wilson"
        },
        {
            "id": 4,
            "user": "Alice Brown",
            "action": "commented on",
            "object": "issue #123",
            "timestamp": datetime.now() - timedelta(hours=2),
            "avatar_url": "https://ui-avatars.com/api/?name=Alice+Brown"
        }
    ]
    
    # Limit the results
    activities = activities[:limit]
    
    return templates.TemplateResponse(
        "components/activity_feed.html",
        {"request": request, "activities": activities}
    )