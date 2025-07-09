#!/usr/bin/env python3
"""
Branch cleanup utility script for manual repository maintenance.

This script provides utilities for analyzing and cleaning up stale branches,
complementing the automated Branch & Stash Cleanup workflow.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

app = typer.Typer(
    help="Branch cleanup utility for manual repository maintenance.",
    add_completion=False
)
console = Console()


class BranchInfo:
    """Information about a Git branch."""
    
    def __init__(self, name: str, last_commit: datetime, is_merged: bool = False):
        self.name = name
        self.last_commit = last_commit
        self.is_merged = is_merged
        self.days_stale = (datetime.now() - last_commit).days
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'last_commit': self.last_commit.isoformat(),
            'days_stale': self.days_stale,
            'is_merged': self.is_merged
        }


def run_git_command(command: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """Run a git command and return success status and output."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()


def get_branch_last_commit_date(branch: str, remote: bool = True) -> datetime:
    """Get the last commit date for a branch."""
    try:
        branch_ref = f'origin/{branch}' if remote else branch
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ct', branch_ref],
            capture_output=True,
            text=True,
            check=True
        )
        timestamp = int(result.stdout.strip())
        return datetime.fromtimestamp(timestamp)
    except (subprocess.CalledProcessError, ValueError):
        return datetime.now()


def get_all_branches(remote: bool = True) -> List[str]:
    """Get all branches (local or remote)."""
    try:
        if remote:
            cmd = ['git', 'branch', '-r', '--format=%(refname:short)']
        else:
            cmd = ['git', 'branch', '--format=%(refname:short)']
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        branches = []
        
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('origin/HEAD'):
                if remote:
                    # Remove 'origin/' prefix for remote branches
                    branch = line.replace('origin/', '')
                    if branch != 'main':  # Skip main branch
                        branches.append(branch)
                else:
                    if line != 'main':  # Skip main branch
                        branches.append(line)
        
        return branches
    except subprocess.CalledProcessError:
        return []


def is_branch_merged(branch: str, target_branch: str = 'main') -> bool:
    """Check if a branch is already merged into the target branch."""
    try:
        result = subprocess.run(
            ['git', 'branch', '-r', '--merged', f'origin/{target_branch}'],
            capture_output=True,
            text=True,
            check=True
        )
        merged_branches = [
            line.strip().replace('origin/', '') 
            for line in result.stdout.strip().split('\n')
            if line.strip() and not line.strip().startswith('origin/HEAD')
        ]
        return branch in merged_branches
    except subprocess.CalledProcessError:
        return False


def analyze_branches(stale_days: int = 30, remote: bool = True) -> List[BranchInfo]:
    """Analyze all branches and return information about them."""
    branches = get_all_branches(remote=remote)
    branch_info = []
    
    console.print(f"🔍 Analyzing {len(branches)} branches...")
    
    for branch in branches:
        last_commit = get_branch_last_commit_date(branch, remote=remote)
        is_merged = is_branch_merged(branch)
        
        branch_info.append(BranchInfo(
            name=branch,
            last_commit=last_commit,
            is_merged=is_merged
        ))
    
    return branch_info


def display_branch_table(branches: List[BranchInfo], title: str = "Branch Analysis"):
    """Display branches in a formatted table."""
    if not branches:
        console.print("No branches found.")
        return
    
    table = Table(title=title)
    table.add_column("Branch", style="cyan")
    table.add_column("Days Stale", style="red")
    table.add_column("Last Commit", style="dim")
    table.add_column("Status", style="green")
    
    for branch in sorted(branches, key=lambda b: b.days_stale, reverse=True):
        status = "🔀 Merged" if branch.is_merged else "🔄 Active"
        table.add_row(
            branch.name,
            str(branch.days_stale),
            branch.last_commit.strftime('%Y-%m-%d'),
            status
        )
    
    console.print(table)


@app.command()
def analyze(
    stale_days: int = typer.Option(30, "--stale-days", help="Days after which a branch is considered stale"),
    remote: bool = typer.Option(True, "--remote/--local", help="Analyze remote or local branches"),
    output: Optional[str] = typer.Option(None, "--output", help="Save analysis to JSON file"),
    show_all: bool = typer.Option(False, "--show-all", help="Show all branches, not just stale ones")
):
    """Analyze branches for staleness and merge status."""
    console.print("[bold blue]🔍 Branch Analysis[/bold blue]")
    console.print("")
    
    # Fetch latest remote info
    if remote:
        console.print("Fetching latest remote information...")
        success, output_msg = run_git_command(['git', 'fetch', '--prune'])
        if not success:
            console.print(f"[yellow]Warning: Failed to fetch remote info: {output_msg}[/yellow]")
    
    # Analyze branches
    all_branches = analyze_branches(stale_days=stale_days, remote=remote)
    
    if show_all:
        display_branch_table(all_branches, "All Branches")
    else:
        stale_branches = [b for b in all_branches if b.days_stale >= stale_days]
        display_branch_table(stale_branches, f"Stale Branches (>{stale_days} days)")
    
    # Show summary
    total_branches = len(all_branches)
    stale_count = len([b for b in all_branches if b.days_stale >= stale_days])
    merged_count = len([b for b in all_branches if b.is_merged])
    
    console.print(f"\n📊 Summary:")
    console.print(f"  • Total branches: {total_branches}")
    console.print(f"  • Stale branches: {stale_count}")
    console.print(f"  • Merged branches: {merged_count}")
    console.print(f"  • Active branches: {total_branches - stale_count}")
    
    # Save to file if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'stale_days_threshold': stale_days,
            'remote_analysis': remote,
            'summary': {
                'total_branches': total_branches,
                'stale_branches': stale_count,
                'merged_branches': merged_count,
                'active_branches': total_branches - stale_count
            },
            'branches': [b.to_dict() for b in all_branches]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        console.print(f"\n💾 Analysis saved to: {output_path}")


@app.command()
def cleanup(
    stale_days: int = typer.Option(30, "--stale-days", help="Days after which a branch is considered stale"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompts"),
    merged_only: bool = typer.Option(False, "--merged-only", help="Only delete merged branches"),
    remote: bool = typer.Option(True, "--remote/--local", help="Clean up remote or local branches")
):
    """Interactive cleanup of stale branches."""
    console.print("[bold red]🧹 Branch Cleanup[/bold red]")
    console.print("")
    
    if dry_run:
        console.print("[yellow]Running in DRY-RUN mode - no changes will be made[/yellow]")
        console.print("")
    
    # Fetch latest remote info
    if remote:
        console.print("Fetching latest remote information...")
        success, output_msg = run_git_command(['git', 'fetch', '--prune'])
        if not success:
            console.print(f"[yellow]Warning: Failed to fetch remote info: {output_msg}[/yellow]")
    
    # Analyze branches
    all_branches = analyze_branches(stale_days=stale_days, remote=remote)
    
    # Filter branches for cleanup
    if merged_only:
        candidates = [b for b in all_branches if b.is_merged and b.days_stale >= stale_days]
        console.print(f"Found {len(candidates)} merged stale branches for cleanup.")
    else:
        candidates = [b for b in all_branches if b.days_stale >= stale_days]
        console.print(f"Found {len(candidates)} stale branches for cleanup.")
    
    if not candidates:
        console.print("✅ No branches found for cleanup!")
        return
    
    # Display candidates
    display_branch_table(candidates, "Branches to Clean Up")
    
    # Confirm cleanup
    if not force and not dry_run:
        if not Confirm.ask(f"\nDo you want to delete these {len(candidates)} branches?"):
            console.print("Cleanup cancelled.")
            return
    
    # Perform cleanup
    deleted_branches = []
    failed_branches = []
    
    for branch in candidates:
        if dry_run:
            console.print(f"[yellow]Would delete:[/yellow] {branch.name}")
            continue
        
        if remote:
            success, output_msg = run_git_command(['git', 'push', 'origin', '--delete', branch.name])
            if success:
                console.print(f"[green]✅ Deleted remote branch:[/green] {branch.name}")
                deleted_branches.append(branch.name)
            else:
                console.print(f"[red]❌ Failed to delete remote branch:[/red] {branch.name} - {output_msg}")
                failed_branches.append(branch.name)
        else:
            success, output_msg = run_git_command(['git', 'branch', '-D', branch.name])
            if success:
                console.print(f"[green]✅ Deleted local branch:[/green] {branch.name}")
                deleted_branches.append(branch.name)
            else:
                console.print(f"[red]❌ Failed to delete local branch:[/red] {branch.name} - {output_msg}")
                failed_branches.append(branch.name)
    
    # Summary
    console.print(f"\n📊 Cleanup Summary:")
    console.print(f"  • Branches deleted: {len(deleted_branches)}")
    console.print(f"  • Branches failed: {len(failed_branches)}")
    
    if failed_branches:
        console.print(f"\n[red]Failed to delete:[/red]")
        for branch in failed_branches:
            console.print(f"  • {branch}")


@app.command()
def stash_check(
    branch: str = typer.Option("main", "--branch", help="Branch to check for stashes"),
    fix: bool = typer.Option(False, "--fix", help="Automatically clear stashes")
):
    """Check for stashes on a specific branch."""
    console.print(f"[bold blue]🗂️  Stash Check for '{branch}' branch[/bold blue]")
    console.print("")
    
    # Switch to the specified branch
    success, output_msg = run_git_command(['git', 'checkout', branch])
    if not success:
        console.print(f"[red]❌ Failed to checkout {branch}: {output_msg}[/red]")
        return
    
    # Check for stashes
    success, stash_list = run_git_command(['git', 'stash', 'list'])
    
    if not success or not stash_list:
        console.print(f"✅ No stashes found on '{branch}' branch")
        return
    
    # Display stashes
    console.print(f"[red]❌ Found stashes on '{branch}' branch:[/red]")
    for i, stash in enumerate(stash_list.split('\n')):
        console.print(f"  {i}: {stash}")
    
    # Handle fix option
    if fix:
        if Confirm.ask(f"\nClear all stashes on '{branch}' branch?"):
            success, output_msg = run_git_command(['git', 'stash', 'clear'])
            if success:
                console.print(f"[green]✅ Cleared all stashes on '{branch}' branch[/green]")
            else:
                console.print(f"[red]❌ Failed to clear stashes: {output_msg}[/red]")
        else:
            console.print("Stash clearing cancelled.")
    else:
        console.print(f"\n💡 To clear stashes, run:")
        console.print("  • git stash clear  (clear all stashes)")
        console.print("  • git stash drop stash@{N}  (drop specific stash)")
        console.print("  • Or use --fix flag to clear automatically")


@app.command()
def report(
    output: str = typer.Option("reports/branch-analysis-report.json", "--output", help="Output file for report"),
    stale_days: int = typer.Option(30, "--stale-days", help="Days after which a branch is considered stale")
):
    """Generate a comprehensive branch analysis report."""
    console.print("[bold blue]📊 Generating Branch Analysis Report[/bold blue]")
    console.print("")
    
    # Fetch latest remote info
    console.print("Fetching latest remote information...")
    success, output_msg = run_git_command(['git', 'fetch', '--prune'])
    if not success:
        console.print(f"[yellow]Warning: Failed to fetch remote info: {output_msg}[/yellow]")
    
    # Analyze branches
    all_branches = analyze_branches(stale_days=stale_days, remote=True)
    
    # Check for stashes on main
    success, stash_list = run_git_command(['git', 'stash', 'list'])
    has_stashes = success and bool(stash_list)
    
    # Generate report
    stale_branches = [b for b in all_branches if b.days_stale >= stale_days]
    merged_branches = [b for b in all_branches if b.is_merged]
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'stale_days_threshold': stale_days,
        'repository_health': {
            'total_branches': len(all_branches),
            'stale_branches': len(stale_branches),
            'merged_branches': len(merged_branches),
            'active_branches': len(all_branches) - len(stale_branches),
            'has_stashes_on_main': has_stashes
        },
        'stale_branches': [b.to_dict() for b in stale_branches],
        'merged_branches': [b.to_dict() for b in merged_branches],
        'all_branches': [b.to_dict() for b in all_branches],
        'stashes_on_main': stash_list.split('\n') if has_stashes else [],
        'recommendations': []
    }
    
    # Add recommendations
    recommendations = []
    if len(stale_branches) > 0:
        recommendations.append(f"Consider cleaning up {len(stale_branches)} stale branches")
    if len(merged_branches) > 0:
        recommendations.append(f"Consider deleting {len(merged_branches)} merged branches")
    if has_stashes:
        recommendations.append("Clean up stashes on main branch")
    if not recommendations:
        recommendations.append("Repository branch hygiene is good!")
    
    report_data['recommendations'] = recommendations
    
    # Save report
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Display summary
    console.print(f"📊 Branch Analysis Summary:")
    console.print(f"  • Total branches: {len(all_branches)}")
    console.print(f"  • Stale branches: {len(stale_branches)}")
    console.print(f"  • Merged branches: {len(merged_branches)}")
    console.print(f"  • Has stashes on main: {has_stashes}")
    console.print(f"\n💾 Report saved to: {output_path}")


if __name__ == "__main__":
    app()
