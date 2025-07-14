"""Form handling endpoints for HTMX forms."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, ValidationError

router = APIRouter()


def get_templates(request: Request) -> Jinja2Templates:
    """Get templates instance from app state."""
    return request.app.state.templates


class ContactForm(BaseModel):
    """Contact form model."""
    name: str
    email: EmailStr
    subject: str
    message: str


class NewsletterForm(BaseModel):
    """Newsletter subscription form model."""
    email: EmailStr


class LoginForm(BaseModel):
    """Login form model."""
    email: EmailStr
    password: str
    remember_me: bool = False


@router.post("/contact")
async def handle_contact_form(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
    templates: Jinja2Templates = Depends(get_templates)
):
    """Handle contact form submission."""
    try:
        # Validate form data
        contact_data = ContactForm(
            name=name,
            email=email,
            subject=subject,
            message=message
        )
        
        # Simulate form processing delay
        await asyncio.sleep(1)
        
        # Here you would typically:
        # - Save to database
        # - Send email notification
        # - Log the contact request
        
        return templates.TemplateResponse(
            "components/form_success.html",
            {
                "request": request,
                "message": f"Thank you, {contact_data.name}! Your message has been sent successfully.",
                "type": "success"
            }
        )
        
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error["loc"][-1]
            msg = error["msg"]
            error_messages.append(f"{field.title()}: {msg}")
        
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "; ".join(error_messages),
                "type": "error"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "An unexpected error occurred. Please try again.",
                "type": "error"
            }
        )


@router.post("/newsletter")
async def handle_newsletter_form(
    request: Request,
    email: str = Form(...),
    templates: Jinja2Templates = Depends(get_templates)
):
    """Handle newsletter subscription."""
    try:
        # Validate email
        newsletter_data = NewsletterForm(email=email)
        
        # Simulate API call to newsletter service
        await asyncio.sleep(1)
        
        # Here you would typically:
        # - Add to newsletter service (e.g., Mailchimp, SendGrid)
        # - Save to database
        # - Send welcome email
        
        return templates.TemplateResponse(
            "components/form_success.html",
            {
                "request": request,
                "message": f"Successfully subscribed {newsletter_data.email} to our newsletter!",
                "type": "success"
            }
        )
        
    except ValidationError as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "Please enter a valid email address.",
                "type": "error"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "Failed to subscribe. Please try again.",
                "type": "error"
            }
        )


@router.post("/login")
async def handle_login_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False),
    templates: Jinja2Templates = Depends(get_templates)
):
    """Handle login form submission."""
    try:
        # Validate form data
        login_data = LoginForm(
            email=email,
            password=password,
            remember_me=remember_me
        )
        
        # Simulate authentication delay
        await asyncio.sleep(1)
        
        # Mock authentication logic
        if email == "demo@example.com" and password == "password":
            # Successful login
            return templates.TemplateResponse(
                "components/form_success.html",
                {
                    "request": request,
                    "message": "Login successful! Redirecting...",
                    "type": "success",
                    "redirect": "/dashboard"
                }
            )
        else:
            # Failed login
            return templates.TemplateResponse(
                "components/form_error.html",
                {
                    "request": request,
                    "error": "Invalid email or password.",
                    "type": "error"
                }
            )
            
    except ValidationError as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "Please check your input and try again.",
                "type": "error"
            }
        )


@router.post("/task")
async def handle_task_form(
    request: Request,
    task_title: str = Form(...),
    task_priority: str = Form("medium"),
    templates: Jinja2Templates = Depends(get_templates)
):
    """Handle new task creation."""
    try:
        if not task_title.strip():
            raise ValueError("Task title is required")
        
        # Simulate saving to database
        await asyncio.sleep(0.5)
        
        # Create new task object
        new_task = {
            "id": 999,  # Would be generated by database
            "title": task_title.strip(),
            "priority": task_priority,
            "completed": False,
            "created_at": "just now"
        }
        
        return templates.TemplateResponse(
            "components/task_item.html",
            {
                "request": request,
                "task": new_task,
                "is_new": True
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "Failed to create task. Please try again.",
                "type": "error"
            }
        )


@router.post("/search")
async def handle_search_form(
    request: Request,
    query: str = Form(...),
    templates: Jinja2Templates = Depends(get_templates)
):
    """Handle search form submission."""
    try:
        if not query.strip():
            return templates.TemplateResponse(
                "components/search_results.html",
                {"request": request, "results": [], "query": ""}
            )
        
        # Simulate search delay
        await asyncio.sleep(0.5)
        
        # Mock search results
        all_results = [
            {"id": 1, "title": "HTMX Documentation", "url": "/docs/htmx", "description": "Complete guide to HTMX"},
            {"id": 2, "title": "Tailwind CSS Guide", "url": "/docs/tailwind", "description": "Styling with Tailwind CSS"},
            {"id": 3, "title": "FastAPI Tutorial", "url": "/docs/fastapi", "description": "Building APIs with FastAPI"},
            {"id": 4, "title": "Form Handling", "url": "/examples/forms", "description": "Dynamic form examples"},
            {"id": 5, "title": "Component Library", "url": "/components", "description": "Reusable UI components"},
        ]
        
        # Filter results based on query
        query_lower = query.lower()
        results = [
            result for result in all_results
            if query_lower in result["title"].lower() or query_lower in result["description"].lower()
        ]
        
        return templates.TemplateResponse(
            "components/search_results.html",
            {
                "request": request,
                "results": results,
                "query": query,
                "count": len(results)
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "Search failed. Please try again.",
                "type": "error"
            }
        )


@router.put("/task/{task_id}/toggle")
async def toggle_task(
    task_id: int,
    request: Request,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Toggle task completion status."""
    try:
        # Simulate database update
        await asyncio.sleep(0.3)
        
        # Mock task data (would come from database)
        task = {
            "id": task_id,
            "title": f"Task {task_id}",
            "priority": "medium",
            "completed": True,  # Toggle the status
            "updated_at": "just now"
        }
        
        return templates.TemplateResponse(
            "components/task_item.html",
            {"request": request, "task": task}
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "Failed to update task.",
                "type": "error"
            }
        )


@router.delete("/task/{task_id}")
async def delete_task(
    task_id: int,
    request: Request,
    templates: Jinja2Templates = Depends(get_templates)
):
    """Delete a task."""
    try:
        # Simulate database deletion
        await asyncio.sleep(0.3)
        
        # Return empty response to remove the element
        return ""
        
    except Exception as e:
        return templates.TemplateResponse(
            "components/form_error.html",
            {
                "request": request,
                "error": "Failed to delete task.",
                "type": "error"
            }
        )