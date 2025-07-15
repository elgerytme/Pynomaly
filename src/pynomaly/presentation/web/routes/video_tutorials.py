"""
Video Tutorial Web Routes

Provides web interface routes for accessing and managing video tutorials.
"""

import asyncio
from pathlib import Path
from typing import Optional
from uuid import UUID

from flask import Blueprint, jsonify, render_template, request

from pynomaly.application.services.video_tutorial_service import VideoTutorialService

# Create blueprint
video_tutorials_bp = Blueprint('video_tutorials', __name__, url_prefix='/tutorials/video')

# Initialize video tutorial service
VIDEO_STORAGE_PATH = Path.home() / ".pynomaly" / "video_tutorials"
video_service = VideoTutorialService(VIDEO_STORAGE_PATH)


@video_tutorials_bp.route('/')
def video_tutorials_home():
    """Main video tutorials page."""
    return render_template('video_tutorials.html')


@video_tutorials_bp.route('/api/series')
def get_video_series():
    """Get all video series."""
    async def _get_series():
        published_only = request.args.get('published_only', 'true').lower() == 'true'
        category = request.args.get('category')
        difficulty = request.args.get('difficulty')
        
        series_list = await video_service.get_video_series(published_only=published_only)
        
        # Apply filters
        if category:
            series_list = [s for s in series_list if s.category == category]
        if difficulty:
            series_list = [s for s in series_list if s.difficulty_level == difficulty]
        
        return [series.dict() for series in series_list]
    
    try:
        series_data = asyncio.run(_get_series())
        return jsonify({
            'success': True,
            'data': series_data,
            'count': len(series_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/series/<series_id>')
def get_series_details(series_id: str):
    """Get detailed information about a specific series."""
    async def _get_series_details():
        try:
            series_uuid = UUID(series_id)
            series = await video_service.get_video_series_by_id(series_uuid)
            
            if not series:
                return None
            
            return series.dict()
        except ValueError:
            return None
    
    try:
        series_data = asyncio.run(_get_series_details())
        
        if series_data is None:
            return jsonify({
                'success': False,
                'error': 'Series not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': series_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/tutorials/<tutorial_id>')
def get_tutorial_details(tutorial_id: str):
    """Get detailed information about a specific tutorial."""
    async def _get_tutorial_details():
        try:
            tutorial_uuid = UUID(tutorial_id)
            tutorial = await video_service.get_video_tutorial(tutorial_uuid)
            
            if not tutorial:
                return None
            
            return tutorial.dict()
        except ValueError:
            return None
    
    try:
        tutorial_data = asyncio.run(_get_tutorial_details())
        
        if tutorial_data is None:
            return jsonify({
                'success': False,
                'error': 'Tutorial not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': tutorial_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/search')
def search_tutorials():
    """Search for tutorials."""
    async def _search_tutorials():
        query = request.args.get('q', '')
        category = request.args.get('category')
        difficulty = request.args.get('difficulty')
        
        if not query:
            return []
        
        results = await video_service.search_tutorials(
            query=query,
            category=category,
            difficulty=difficulty
        )
        
        return [tutorial.dict() for tutorial in results]
    
    try:
        results = asyncio.run(_search_tutorials())
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/progress/<user_id>')
def get_user_progress(user_id: str):
    """Get user's learning progress."""
    async def _get_progress():
        series_id = request.args.get('series_id')
        
        if series_id:
            try:
                series_uuid = UUID(series_id)
                progress = await video_service.get_user_progress_for_series(user_id, series_uuid)
            except ValueError:
                return None
        else:
            progress = await video_service.get_user_progress(user_id)
        
        return [p.dict() for p in progress]
    
    try:
        progress_data = asyncio.run(_get_progress())
        
        if progress_data is None:
            return jsonify({
                'success': False,
                'error': 'Invalid series ID'
            }), 400
        
        return jsonify({
            'success': True,
            'data': progress_data,
            'count': len(progress_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/track-view', methods=['POST'])
def track_video_view():
    """Track video view and engagement."""
    async def _track_view():
        data = request.get_json()
        
        tutorial_id = data.get('tutorial_id')
        user_id = data.get('user_id', 'guest')
        watch_time = data.get('watch_time', 0)
        device_type = data.get('device_type', 'desktop')
        
        if not tutorial_id:
            return False
        
        try:
            tutorial_uuid = UUID(tutorial_id)
            await video_service.track_video_view(
                tutorial_id=tutorial_uuid,
                user_id=user_id,
                watch_time=watch_time,
                device_type=device_type
            )
            return True
        except ValueError:
            return False
    
    try:
        success = asyncio.run(_track_view())
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Invalid tutorial ID or missing required fields'
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Video view tracked successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/bookmark', methods=['POST'])
def manage_bookmark():
    """Add or remove a bookmark."""
    async def _manage_bookmark():
        data = request.get_json()
        
        tutorial_id = data.get('tutorial_id')
        user_id = data.get('user_id', 'guest')
        timestamp = data.get('timestamp')
        
        if not tutorial_id or timestamp is None:
            return False
        
        try:
            tutorial_uuid = UUID(tutorial_id)
            await video_service.add_bookmark(user_id, tutorial_uuid, timestamp)
            return True
        except ValueError:
            return False
    
    try:
        success = asyncio.run(_manage_bookmark())
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Invalid tutorial ID or missing required fields'
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Bookmark added successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/rate', methods=['POST'])
def rate_tutorial():
    """Rate a tutorial and provide feedback."""
    async def _rate_tutorial():
        data = request.get_json()
        
        tutorial_id = data.get('tutorial_id')
        user_id = data.get('user_id', 'guest')
        rating = data.get('rating')
        feedback = data.get('feedback')
        
        if not tutorial_id or rating is None:
            return False
        
        try:
            tutorial_uuid = UUID(tutorial_id)
            await video_service.rate_tutorial(user_id, tutorial_uuid, rating, feedback)
            return True
        except ValueError:
            return False
    
    try:
        success = asyncio.run(_rate_tutorial())
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Invalid tutorial ID or missing required fields'
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Rating submitted successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/recommendations/<user_id>')
def get_recommendations(user_id: str):
    """Get personalized tutorial recommendations."""
    async def _get_recommendations():
        limit = request.args.get('limit', 5, type=int)
        recommendations = await video_service.get_recommended_tutorials(user_id, limit)
        return [tutorial.dict() for tutorial in recommendations]
    
    try:
        recommendations = asyncio.run(_get_recommendations())
        return jsonify({
            'success': True,
            'data': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/analytics')
def get_analytics():
    """Get learning analytics."""
    async def _get_analytics():
        user_id = request.args.get('user_id')
        tutorial_id = request.args.get('tutorial_id')
        
        if tutorial_id:
            try:
                tutorial_uuid = UUID(tutorial_id)
                analytics = await video_service.get_tutorial_analytics(tutorial_uuid)
                return analytics.dict() if analytics else None
            except ValueError:
                return None
        else:
            analytics = await video_service.get_learning_analytics(user_id)
            return analytics
    
    try:
        analytics_data = asyncio.run(_get_analytics())
        
        if analytics_data is None:
            return jsonify({
                'success': False,
                'error': 'Analytics not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': analytics_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/api/certificate/<user_id>/<series_id>')
def generate_certificate(user_id: str, series_id: str):
    """Generate completion certificate for a series."""
    async def _generate_certificate():
        try:
            series_uuid = UUID(series_id)
            certificate_id = await video_service.generate_certificate(user_id, series_uuid)
            return certificate_id
        except ValueError:
            return None
    
    try:
        certificate_id = asyncio.run(_generate_certificate())
        
        if certificate_id is None:
            return jsonify({
                'success': False,
                'error': 'Invalid series ID or series not completed'
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'certificate_id': certificate_id,
                'message': 'Certificate generated successfully'
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@video_tutorials_bp.route('/player/<tutorial_id>')
def video_player(tutorial_id: str):
    """Video player page for a specific tutorial."""
    async def _get_tutorial():
        try:
            tutorial_uuid = UUID(tutorial_id)
            tutorial = await video_service.get_video_tutorial(tutorial_uuid)
            return tutorial.dict() if tutorial else None
        except ValueError:
            return None
    
    try:
        tutorial_data = asyncio.run(_get_tutorial())
        
        if tutorial_data is None:
            return "Tutorial not found", 404
        
        return render_template('video_player.html', tutorial=tutorial_data)
    except Exception as e:
        return f"Error loading tutorial: {str(e)}", 500


# Error handlers
@video_tutorials_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404


@video_tutorials_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500