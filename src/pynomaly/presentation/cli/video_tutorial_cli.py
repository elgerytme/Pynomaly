"""
Video Tutorial CLI Commands

Provides command-line interface for managing and accessing video tutorials.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from pynomaly.application.services.video_tutorial_service import VideoTutorialService

console = Console()
app = typer.Typer(name="video", help="Video tutorial management and learning tools")

# Initialize video tutorial service
VIDEO_STORAGE_PATH = Path.home() / ".pynomaly" / "video_tutorials"
video_service = VideoTutorialService(VIDEO_STORAGE_PATH)


@app.command("list")
def list_video_series(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    difficulty: Optional[str] = typer.Option(None, "--difficulty", "-d", help="Filter by difficulty level"),
    published_only: bool = typer.Option(True, "--published-only/--all", help="Show only published series")
):
    """List all available video tutorial series."""
    
    async def _list_series():
        series_list = await video_service.get_video_series(published_only=published_only)
        
        # Apply filters
        if category:
            series_list = [s for s in series_list if s.category == category]
        if difficulty:
            series_list = [s for s in series_list if s.difficulty_level == difficulty]
        
        if not series_list:
            console.print("[yellow]No video series found matching your criteria.[/yellow]")
            return
        
        # Create table
        table = Table(title="📚 Video Tutorial Series")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="bold green")
        table.add_column("Category", style="blue")
        table.add_column("Difficulty", style="magenta")
        table.add_column("Videos", justify="center")
        table.add_column("Duration", justify="center")
        table.add_column("Status", justify="center")
        
        for series in series_list:
            status = "✅ Published" if series.is_published else "🚧 Draft"
            table.add_row(
                str(series.id)[:8],
                series.name,
                series.category.title(),
                series.difficulty_level.title(),
                str(len(series.videos)),
                f"{series.total_duration_minutes} min",
                status
            )
        
        console.print(table)
        console.print(f"\n[dim]Found {len(series_list)} video series[/dim]")
    
    asyncio.run(_list_series())


@app.command("show")
def show_series_details(
    series_id: str = typer.Argument(..., help="Series ID to display"),
    show_videos: bool = typer.Option(True, "--videos/--no-videos", help="Show individual videos")
):
    """Show detailed information about a video series."""
    
    async def _show_details():
        try:
            series_uuid = UUID(series_id)
        except ValueError:
            console.print(f"[red]Invalid series ID: {series_id}[/red]")
            return
        
        series = await video_service.get_video_series_by_id(series_uuid)
        if not series:
            console.print(f"[red]Series not found: {series_id}[/red]")
            return
        
        # Series overview
        overview_text = f"""
[bold green]{series.name}[/bold green]
[dim]{series.description}[/dim]

[bold]Category:[/bold] {series.category.title()}
[bold]Difficulty:[/bold] {series.difficulty_level.title()}
[bold]Total Duration:[/bold] {series.total_duration_minutes} minutes
[bold]Videos:[/bold] {len(series.videos)}
[bold]Status:[/bold] {'✅ Published' if series.is_published else '🚧 Draft'}
"""
        
        if series.prerequisites:
            overview_text += f"\n[bold]Prerequisites:[/bold]\n"
            for prereq in series.prerequisites:
                overview_text += f"  • {prereq}\n"
        
        if series.learning_outcomes:
            overview_text += f"\n[bold]Learning Outcomes:[/bold]\n"
            for outcome in series.learning_outcomes:
                overview_text += f"  • {outcome}\n"
        
        console.print(Panel(overview_text, title="📚 Series Overview"))
        
        if show_videos and series.videos:
            # Videos table
            table = Table(title="🎬 Videos in Series")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Title", style="bold")
            table.add_column("Duration", justify="center", width=10)
            table.add_column("Topics", style="dim")
            
            for video in sorted(series.videos, key=lambda x: x.episode_number):
                topics_str = ", ".join(video.topics[:3])
                if len(video.topics) > 3:
                    topics_str += f" (+{len(video.topics) - 3} more)"
                
                table.add_row(
                    str(video.episode_number),
                    video.title,
                    f"{video.duration_minutes} min",
                    topics_str
                )
            
            console.print(table)
    
    asyncio.run(_show_details())


@app.command("watch")
def watch_video(
    tutorial_id: str = typer.Argument(..., help="Tutorial ID to watch"),
    user_id: str = typer.Option("guest", "--user", "-u", help="User ID for progress tracking"),
    quality: str = typer.Option("720p", "--quality", "-q", help="Video quality preference")
):
    """Open a video tutorial for watching."""
    
    async def _watch_video():
        try:
            tutorial_uuid = UUID(tutorial_id)
        except ValueError:
            console.print(f"[red]Invalid tutorial ID: {tutorial_id}[/red]")
            return
        
        tutorial = await video_service.get_video_tutorial(tutorial_uuid)
        if not tutorial:
            console.print(f"[red]Tutorial not found: {tutorial_id}[/red]")
            return
        
        # Display video information
        video_info = f"""
[bold green]{tutorial.title}[/bold green]
[dim]{tutorial.description}[/dim]

[bold]Series:[/bold] {tutorial.series}
[bold]Episode:[/bold] #{tutorial.episode_number}
[bold]Duration:[/bold] {tutorial.duration_minutes} minutes
[bold]Difficulty:[/bold] {tutorial.difficulty_level.title()}
[bold]Rating:[/bold] ⭐ {tutorial.average_rating:.1f}/5.0
"""
        
        if tutorial.learning_objectives:
            video_info += f"\n[bold]Learning Objectives:[/bold]\n"
            for obj in tutorial.learning_objectives:
                video_info += f"  • {obj}\n"
        
        console.print(Panel(video_info, title="🎬 Now Playing"))
        
        # Simulate opening video (in real implementation, this would open browser/player)
        console.print(f"\n[blue]Opening video in browser...[/blue]")
        console.print(f"[dim]URL: {tutorial.video_url}[/dim]")
        console.print(f"[dim]Quality: {quality}[/dim]")
        
        # Track the view
        await video_service.track_video_view(
            tutorial_id=tutorial_uuid,
            user_id=user_id,
            watch_time=60,  # Simulate 1 minute watch time
            device_type="desktop"
        )
        
        console.print(f"\n[green]✅ Video view tracked for user: {user_id}[/green]")
    
    asyncio.run(_watch_video())


@app.command("search")
def search_tutorials(
    query: str = typer.Argument(..., help="Search query"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    difficulty: Optional[str] = typer.Option(None, "--difficulty", "-d", help="Filter by difficulty")
):
    """Search for video tutorials."""
    
    async def _search():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching tutorials...", total=None)
            
            results = await video_service.search_tutorials(
                query=query,
                category=category,
                difficulty=difficulty
            )
            
            progress.update(task, completed=True)
        
        if not results:
            console.print(f"[yellow]No tutorials found for query: '{query}'[/yellow]")
            return
        
        # Display search results
        table = Table(title=f"🔍 Search Results for '{query}'")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="bold")
        table.add_column("Series", style="blue")
        table.add_column("Duration", justify="center")
        table.add_column("Rating", justify="center")
        table.add_column("Topics", style="dim")
        
        for tutorial in results:
            topics_str = ", ".join(tutorial.topics[:2])
            if len(tutorial.topics) > 2:
                topics_str += "..."
            
            table.add_row(
                str(tutorial.id)[:8],
                tutorial.title,
                tutorial.series,
                f"{tutorial.duration_minutes} min",
                f"⭐ {tutorial.average_rating:.1f}",
                topics_str
            )
        
        console.print(table)
        console.print(f"\n[dim]Found {len(results)} tutorials[/dim]")
    
    asyncio.run(_search())


@app.command("progress")
def show_user_progress(
    user_id: str = typer.Option("guest", "--user", "-u", help="User ID to check progress"),
    series_id: Optional[str] = typer.Option(None, "--series", "-s", help="Show progress for specific series")
):
    """Show user's learning progress."""
    
    async def _show_progress():
        if series_id:
            try:
                series_uuid = UUID(series_id)
                progress_list = await video_service.get_user_progress_for_series(user_id, series_uuid)
                title = f"📊 Progress for Series: {series_id[:8]}"
            except ValueError:
                console.print(f"[red]Invalid series ID: {series_id}[/red]")
                return
        else:
            progress_list = await video_service.get_user_progress(user_id)
            title = f"📊 Learning Progress for User: {user_id}"
        
        if not progress_list:
            console.print(f"[yellow]No progress found for user: {user_id}[/yellow]")
            return
        
        # Progress table
        table = Table(title=title)
        table.add_column("Tutorial", style="bold", no_wrap=True)
        table.add_column("Progress", justify="center", width=15)
        table.add_column("Watch Time", justify="center")
        table.add_column("Quiz Scores", justify="center")
        table.add_column("Status", justify="center")
        
        total_watch_time = 0
        completed_count = 0
        
        for progress in progress_list:
            tutorial = await video_service.get_video_tutorial(progress.tutorial_id)
            tutorial_name = tutorial.title if tutorial else str(progress.tutorial_id)[:8]
            
            # Progress bar
            progress_percent = int(progress.completion_percentage)
            progress_bar = "█" * (progress_percent // 10) + "░" * (10 - progress_percent // 10)
            progress_text = f"{progress_bar} {progress_percent}%"
            
            # Quiz scores
            avg_quiz = sum(progress.quiz_scores) / len(progress.quiz_scores) if progress.quiz_scores else 0
            quiz_text = f"{avg_quiz:.1f}%" if progress.quiz_scores else "N/A"
            
            # Status
            if progress.completed_at:
                status = "✅ Completed"
                completed_count += 1
            elif progress.completion_percentage > 0:
                status = "🔄 In Progress"
            else:
                status = "⏳ Not Started"
            
            total_watch_time += progress.watch_time_seconds
            
            table.add_row(
                tutorial_name,
                progress_text,
                f"{progress.watch_time_seconds // 60} min",
                quiz_text,
                status
            )
        
        console.print(table)
        
        # Summary
        summary = f"""
[bold]Summary:[/bold]
• Total Tutorials: {len(progress_list)}
• Completed: {completed_count}
• Total Watch Time: {total_watch_time // 3600}h {(total_watch_time % 3600) // 60}m
• Completion Rate: {(completed_count / len(progress_list) * 100):.1f}%
"""
        console.print(Panel(summary, title="📈 Progress Summary"))
    
    asyncio.run(_show_progress())


@app.command("bookmark")
def manage_bookmarks(
    tutorial_id: str = typer.Argument(..., help="Tutorial ID"),
    user_id: str = typer.Option("guest", "--user", "-u", help="User ID"),
    timestamp: Optional[int] = typer.Option(None, "--time", "-t", help="Timestamp to bookmark (seconds)"),
    action: str = typer.Option("add", "--action", "-a", help="Action: add or list")
):
    """Manage video bookmarks."""
    
    async def _manage_bookmarks():
        try:
            tutorial_uuid = UUID(tutorial_id)
        except ValueError:
            console.print(f"[red]Invalid tutorial ID: {tutorial_id}[/red]")
            return
        
        if action == "add":
            if timestamp is None:
                console.print("[red]Timestamp is required for adding bookmarks[/red]")
                return
            
            await video_service.add_bookmark(user_id, tutorial_uuid, timestamp)
            console.print(f"[green]✅ Bookmark added at {timestamp // 60}:{timestamp % 60:02d}[/green]")
        
        elif action == "list":
            progress_list = await video_service.get_user_progress(user_id)
            for progress in progress_list:
                if progress.tutorial_id == tutorial_uuid:
                    if progress.bookmarks:
                        console.print(f"[bold]Bookmarks for tutorial {tutorial_id[:8]}:[/bold]")
                        for i, bookmark in enumerate(progress.bookmarks, 1):
                            minutes, seconds = divmod(bookmark, 60)
                            console.print(f"  {i}. {minutes}:{seconds:02d}")
                    else:
                        console.print("[yellow]No bookmarks found for this tutorial[/yellow]")
                    return
            
            console.print("[yellow]No progress found for this tutorial[/yellow]")
    
    asyncio.run(_manage_bookmarks())


@app.command("analytics")
def show_analytics(
    user_id: Optional[str] = typer.Option(None, "--user", "-u", help="User ID for personal analytics"),
    tutorial_id: Optional[str] = typer.Option(None, "--tutorial", "-t", help="Tutorial ID for specific analytics")
):
    """Show learning analytics and statistics."""
    
    async def _show_analytics():
        if tutorial_id:
            # Tutorial-specific analytics
            try:
                tutorial_uuid = UUID(tutorial_id)
                analytics = await video_service.get_tutorial_analytics(tutorial_uuid)
                
                if not analytics:
                    console.print(f"[red]No analytics found for tutorial: {tutorial_id}[/red]")
                    return
                
                analytics_text = f"""
[bold]Tutorial Analytics[/bold]

[bold]Engagement:[/bold]
• Total Views: {analytics.total_views:,}
• Unique Viewers: {analytics.unique_viewers:,}
• Total Watch Time: {analytics.total_watch_time // 3600}h {(analytics.total_watch_time % 3600) // 60}m
• Average Watch Time: {analytics.average_watch_time:.1f} seconds
• Completion Rate: {analytics.completion_rate:.1f}%

[bold]Performance:[/bold]
• Average Loading Time: {sum(analytics.loading_times) / len(analytics.loading_times) if analytics.loading_times else 0:.2f}s
• Error Events: {len(analytics.error_events)}
"""
                
                if analytics.device_stats:
                    analytics_text += f"\n[bold]Device Distribution:[/bold]\n"
                    for device, count in analytics.device_stats.items():
                        analytics_text += f"• {device.title()}: {count}\n"
                
                console.print(Panel(analytics_text, title="📊 Tutorial Analytics"))
                
            except ValueError:
                console.print(f"[red]Invalid tutorial ID: {tutorial_id}[/red]")
        
        else:
            # User or system-wide analytics
            analytics = await video_service.get_learning_analytics(user_id)
            
            if user_id:
                analytics_text = f"""
[bold]Personal Learning Analytics[/bold]

[bold]Progress:[/bold]
• Tutorials Started: {analytics['tutorials_started']}
• Tutorials Completed: {analytics['tutorials_completed']}
• Total Watch Time: {analytics['total_watch_time_minutes']:.1f} minutes
• Average Completion Rate: {analytics['average_completion_rate']:.1f}%
• Certificates Earned: {analytics['certificates_earned']}

[bold]Favorite Topics:[/bold]
"""
                for topic in analytics['favorite_topics']:
                    analytics_text += f"• {topic}\n"
                
                title = f"👤 Analytics for {user_id}"
            
            else:
                analytics_text = f"""
[bold]System-Wide Analytics[/bold]

[bold]Content:[/bold]
• Total Tutorials: {analytics['total_tutorials']}
• Total Series: {analytics['total_series']}
• Total Users: {analytics['total_users']}
• Total Views: {analytics['total_views']:,}
• Total Watch Time: {analytics['total_watch_time_hours']:.1f} hours

[bold]Engagement:[/bold]
• Average Completion Rate: {analytics['average_completion_rate']:.1f}%

[bold]Popular Tutorials:[/bold]
"""
                for tutorial in analytics['popular_tutorials'][:5]:
                    analytics_text += f"• {tutorial['title']} ({tutorial['views']} views)\n"
                
                analytics_text += f"\n[bold]Trending Topics:[/bold]\n"
                for topic in analytics['trending_topics'][:5]:
                    analytics_text += f"• {topic}\n"
                
                title = "🌍 Global Analytics"
            
            console.print(Panel(analytics_text, title=title))
    
    asyncio.run(_show_analytics())


@app.command("recommend")
def get_recommendations(
    user_id: str = typer.Option("guest", "--user", "-u", help="User ID for personalized recommendations"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of recommendations")
):
    """Get personalized tutorial recommendations."""
    
    async def _get_recommendations():
        recommendations = await video_service.get_recommended_tutorials(user_id, limit)
        
        if not recommendations:
            console.print(f"[yellow]No recommendations available for user: {user_id}[/yellow]")
            return
        
        console.print(f"[bold green]🎯 Recommended Tutorials for {user_id}[/bold green]\n")
        
        for i, tutorial in enumerate(recommendations, 1):
            recommendation_text = f"""
[bold]{i}. {tutorial.title}[/bold]
   [dim]{tutorial.description}[/dim]
   
   [blue]Series:[/blue] {tutorial.series}
   [blue]Duration:[/blue] {tutorial.duration_minutes} minutes
   [blue]Difficulty:[/blue] {tutorial.difficulty_level.title()}
   [blue]Rating:[/blue] ⭐ {tutorial.average_rating:.1f}/5.0
   [blue]Views:[/blue] {tutorial.view_count:,}
   
   [green]Learning Objectives:[/green]
"""
            for obj in tutorial.learning_objectives[:2]:
                recommendation_text += f"   • {obj}\n"
            
            if len(tutorial.learning_objectives) > 2:
                recommendation_text += f"   • ... and {len(tutorial.learning_objectives) - 2} more\n"
            
            console.print(Panel(recommendation_text, border_style="blue"))
    
    asyncio.run(_get_recommendations())


if __name__ == "__main__":
    app()