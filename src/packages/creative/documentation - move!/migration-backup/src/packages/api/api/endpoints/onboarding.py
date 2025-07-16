"""
API endpoints for user onboarding system.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from monorepo.infrastructure.auth.enhanced_dependencies import get_current_user
from monorepo.presentation.web.onboarding import (
    ExperienceLevel,
    OnboardingGoal,
    OnboardingStage,
    UserRole,
    onboarding_service,
)

router = APIRouter(prefix="/api/v1/onboarding", tags=["Onboarding"])


class StartOnboardingRequest(BaseModel):
    """Request to start user onboarding."""

    user_id: str | None = None
    skip_intro: bool = False


class StartOnboardingResponse(BaseModel):
    """Response for starting onboarding."""

    onboarding_id: str
    current_step: dict[str, Any]
    estimated_total_time: int
    stages: list[str]


class UpdateProfileRequest(BaseModel):
    """Request to update user profile."""

    role: UserRole
    experience_level: ExperienceLevel
    goals: list[OnboardingGoal]
    programming_languages: list[str] = Field(default_factory=list)
    ml_frameworks: list[str] = Field(default_factory=list)
    industry: str | None = None
    use_case: str | None = None
    preferred_learning_style: str = "interactive"


class CompleteStepRequest(BaseModel):
    """Request to complete a step."""

    step_id: str
    completion_time: int | None = None  # seconds taken
    feedback: dict[str, Any] | None = None
    satisfaction_rating: int | None = Field(None, ge=1, le=5)
    comments: str | None = None


class OnboardingProgressResponse(BaseModel):
    """Response with onboarding progress."""

    user_id: str
    role: str
    experience_level: str
    current_stage: str
    completion_ratio: float
    completed_steps: int
    total_steps: int
    estimated_time_remaining: int
    achievements: list[str]
    started_at: str
    last_activity: str
    next_step: str | None


class OnboardingStepResponse(BaseModel):
    """Response with step details."""

    id: str
    title: str
    description: str
    content: str
    estimated_time: int
    difficulty: str
    required: bool
    prerequisites: list[str]
    resources: list[dict[str, str]]
    interactive_elements: list[dict[str, Any]]
    progress_indicator: dict[str, Any]


class PersonalizedResourceResponse(BaseModel):
    """Response with personalized resources."""

    resources: list[dict[str, Any]]
    recommended_next_steps: list[str]
    role_specific_tips: list[str]


@router.post("/start", response_model=StartOnboardingResponse)
async def start_onboarding(
    request: StartOnboardingRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Start the onboarding process for a user."""
    user_id = request.user_id or current_user.get("user_id", "anonymous")

    try:
        progress = onboarding_service.start_onboarding(user_id, http_request)

        # Get first step
        first_step = onboarding_service.get_next_step(user_id)
        if not first_step:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize onboarding",
            )

        # Calculate total estimated time
        estimated_total_time = sum(
            step.estimated_time for step in onboarding_service.steps_db.values()
        )

        return StartOnboardingResponse(
            onboarding_id=progress.user_id,
            current_step={
                "id": first_step.id,
                "title": first_step.title,
                "description": first_step.description,
                "content": first_step.content,
                "estimated_time": first_step.estimated_time,
                "difficulty": first_step.difficulty,
            },
            estimated_total_time=estimated_total_time,
            stages=[stage.value for stage in OnboardingStage],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start onboarding: {str(e)}",
        )


@router.post("/profile")
async def update_profile(
    request: UpdateProfileRequest, current_user: dict = Depends(get_current_user)
):
    """Update user profile for personalized onboarding."""
    user_id = current_user.get("user_id", "anonymous")

    try:
        profile_data = request.dict()
        profile = onboarding_service.update_user_profile(user_id, profile_data)

        return {
            "message": "Profile updated successfully",
            "user_id": user_id,
            "role": profile.role.value,
            "experience_level": profile.experience_level.value,
            "personalized_path_created": True,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}",
        )


@router.get("/step/{step_id}", response_model=OnboardingStepResponse)
async def get_step(step_id: str, current_user: dict = Depends(get_current_user)):
    """Get details for a specific onboarding step."""
    user_id = current_user.get("user_id", "anonymous")

    step = onboarding_service.steps_db.get(step_id)
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Step '{step_id}' not found"
        )

    # Get progress for progress indicator
    progress_summary = onboarding_service.get_progress_summary(user_id)

    progress_indicator = {
        "current_step_number": len(progress_summary.get("completed_steps", [])) + 1,
        "total_steps": progress_summary.get("total_steps", 1),
        "completion_ratio": progress_summary.get("completion_ratio", 0),
        "stage": progress_summary.get("current_stage", "welcome"),
    }

    return OnboardingStepResponse(
        id=step.id,
        title=step.title,
        description=step.description,
        content=step.content,
        estimated_time=step.estimated_time,
        difficulty=step.difficulty,
        required=step.required,
        prerequisites=step.prerequisites,
        resources=step.resources,
        interactive_elements=step.interactive_elements,
        progress_indicator=progress_indicator,
    )


@router.post("/step/complete")
async def complete_step(
    request: CompleteStepRequest, current_user: dict = Depends(get_current_user)
):
    """Mark a step as completed and get next step."""
    user_id = current_user.get("user_id", "anonymous")

    try:
        # Prepare feedback
        feedback = {}
        if request.feedback:
            feedback.update(request.feedback)
        if request.satisfaction_rating:
            feedback["satisfaction_rating"] = request.satisfaction_rating
        if request.comments:
            feedback["comments"] = request.comments
        if request.completion_time:
            feedback["completion_time"] = request.completion_time

        # Complete the step
        success = onboarding_service.complete_step(user_id, request.step_id, feedback)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to complete step",
            )

        # Get next step
        next_step = onboarding_service.get_next_step(user_id)
        progress_summary = onboarding_service.get_progress_summary(user_id)

        response = {
            "message": "Step completed successfully",
            "step_completed": request.step_id,
            "progress": progress_summary,
        }

        if next_step:
            response["next_step"] = {
                "id": next_step.id,
                "title": next_step.title,
                "description": next_step.description,
                "estimated_time": next_step.estimated_time,
            }
        else:
            response["message"] = "Onboarding completed! 🎉"
            response["completion_certificate"] = {
                "user_id": user_id,
                "completed_at": datetime.now().isoformat(),
                "role": progress_summary.get("role"),
                "achievements": progress_summary.get("achievements", []),
            }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete step: {str(e)}",
        )


@router.get("/progress", response_model=OnboardingProgressResponse)
async def get_progress(current_user: dict = Depends(get_current_user)):
    """Get current onboarding progress."""
    user_id = current_user.get("user_id", "anonymous")

    progress_summary = onboarding_service.get_progress_summary(user_id)

    if not progress_summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No onboarding progress found. Please start onboarding first.",
        )

    return OnboardingProgressResponse(**progress_summary)


@router.get("/next-step")
async def get_next_step(current_user: dict = Depends(get_current_user)):
    """Get the next step in the onboarding process."""
    user_id = current_user.get("user_id", "anonymous")

    next_step = onboarding_service.get_next_step(user_id)

    if not next_step:
        return {"message": "No more steps available", "onboarding_complete": True}

    return {
        "step": {
            "id": next_step.id,
            "title": next_step.title,
            "description": next_step.description,
            "content": next_step.content,
            "estimated_time": next_step.estimated_time,
            "difficulty": next_step.difficulty,
            "required": next_step.required,
            "interactive_elements": next_step.interactive_elements,
        }
    }


@router.get("/resources", response_model=PersonalizedResourceResponse)
async def get_personalized_resources(current_user: dict = Depends(get_current_user)):
    """Get personalized learning resources based on user profile."""
    user_id = current_user.get("user_id", "anonymous")

    resources = onboarding_service.get_personalized_resources(user_id)

    # Get role-specific tips
    profile = onboarding_service.user_profiles.get(user_id)
    role_tips = []

    if profile:
        role_tips_map = {
            UserRole.DATA_SCIENTIST: [
                "Focus on algorithm selection and hyperparameter tuning",
                "Experiment with different contamination rates",
                "Use cross-validation for robust model evaluation",
                "Consider ensemble methods for better performance",
            ],
            UserRole.ML_ENGINEER: [
                "Design for scalability from the start",
                "Implement proper monitoring and alerting",
                "Use feature stores for consistent data access",
                "Plan for model versioning and rollback strategies",
            ],
            UserRole.BUSINESS_ANALYST: [
                "Focus on interpretability and explainability",
                "Create clear visualizations for stakeholders",
                "Document business impact and ROI",
                "Establish clear metrics for success",
            ],
            UserRole.DEVOPS_ENGINEER: [
                "Containerize everything for consistency",
                "Implement blue-green deployments",
                "Use infrastructure as code",
                "Monitor resource usage and costs",
            ],
        }
        role_tips = role_tips_map.get(profile.role, [])

    # Get recommended next steps
    progress = onboarding_service.user_progress.get(user_id)
    recommended_steps = []

    if progress and profile:
        learning_path = onboarding_service.learning_paths.get(profile.role, [])
        remaining_steps = [
            step_id
            for step_id in learning_path
            if step_id not in progress.completed_steps
        ]

        # Get the next 3 recommended steps
        for step_id in remaining_steps[:3]:
            step = onboarding_service.steps_db.get(step_id)
            if step:
                recommended_steps.append(
                    {
                        "id": step.id,
                        "title": step.title,
                        "description": step.description,
                        "estimated_time": step.estimated_time,
                    }
                )

    return PersonalizedResourceResponse(
        resources=resources,
        recommended_next_steps=[step["title"] for step in recommended_steps],
        role_specific_tips=role_tips,
    )


@router.post("/skip-step/{step_id}")
async def skip_step(
    step_id: str,
    reason: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Skip a non-required step."""
    user_id = current_user.get("user_id", "anonymous")

    step = onboarding_service.steps_db.get(step_id)
    if not step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Step '{step_id}' not found"
        )

    if step.required:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot skip required steps"
        )

    progress = onboarding_service.user_progress.get(user_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No onboarding progress found"
        )

    # Mark step as skipped
    if step_id not in progress.skipped_steps:
        progress.skipped_steps.append(step_id)

    # Add feedback about why it was skipped
    if reason:
        feedback_entry = {
            "step_id": step_id,
            "action": "skipped",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        progress.feedback.append(feedback_entry)

    # Get next step
    next_step = onboarding_service.get_next_step(user_id)

    response = {"message": f"Step '{step.title}' skipped", "step_skipped": step_id}

    if next_step:
        response["next_step"] = {
            "id": next_step.id,
            "title": next_step.title,
            "description": next_step.description,
        }

    return response


@router.get("/achievements")
async def get_achievements(current_user: dict = Depends(get_current_user)):
    """Get user's onboarding achievements."""
    user_id = current_user.get("user_id", "anonymous")

    progress = onboarding_service.user_progress.get(user_id)
    if not progress:
        return {"achievements": []}

    # Achievement descriptions
    achievement_info = {
        "first_detector": {
            "title": "First Detector",
            "description": "Completed your first anomaly detection",
            "icon": "🎯",
            "points": 10,
        },
        "speed_learner": {
            "title": "Speed Learner",
            "description": "Completed 5 steps in under 30 minutes",
            "icon": "⚡",
            "points": 25,
        },
        "completionist": {
            "title": "Completionist",
            "description": "Completed 80% of your learning path",
            "icon": "🏆",
            "points": 50,
        },
        "explorer": {
            "title": "Explorer",
            "description": "Tried multiple algorithms",
            "icon": "🔍",
            "points": 15,
        },
        "production_ready": {
            "title": "Production Ready",
            "description": "Completed production deployment steps",
            "icon": "🚀",
            "points": 100,
        },
    }

    achievements_with_info = []
    total_points = 0

    for achievement_id in progress.achievements:
        if achievement_id in achievement_info:
            info = achievement_info[achievement_id]
            achievements_with_info.append(
                {
                    "id": achievement_id,
                    **info,
                    "earned_at": progress.started_at.isoformat(),  # Simplified
                }
            )
            total_points += info["points"]

    return {
        "achievements": achievements_with_info,
        "total_points": total_points,
        "next_milestone": "production_ready"
        if "production_ready" not in progress.achievements
        else None,
    }


@router.post("/feedback")
async def submit_feedback(
    feedback: dict[str, Any], current_user: dict = Depends(get_current_user)
):
    """Submit general feedback about the onboarding experience."""
    user_id = current_user.get("user_id", "anonymous")

    progress = onboarding_service.user_progress.get(user_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No onboarding progress found"
        )

    # Add general feedback
    feedback_entry = {
        "type": "general_feedback",
        "feedback": feedback,
        "timestamp": datetime.now().isoformat(),
        "stage": progress.current_stage.value,
    }

    progress.feedback.append(feedback_entry)

    return {
        "message": "Feedback submitted successfully",
        "feedback_id": len(progress.feedback),
    }
