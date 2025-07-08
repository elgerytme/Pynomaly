#!/usr/bin/env python3
"""
Parse docs/project/TODO.md into structured JSON objects.

This script reads the TODO.md file and extracts tasks with their metadata,
converting them into a structured JSON format with priority mapping.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_estimate(estimate_str: str) -> int:
    """Extract numeric estimate from estimate string."""
    match = re.search(r'(\d+)', estimate_str)
    return int(match.group(1)) if match else 0


def map_priority_to_milestone(priority: str) -> str:
    """Map priority levels to P1-P5 milestones."""
    priority_mapping = {
        'Critical': 'P1',
        'High': 'P2',
        'Medium': 'P3',
        'Low': 'P4',
        'informational': 'P5'
    }
    return priority_mapping.get(priority, 'P3')


def extract_layer_from_id(task_id: str) -> str:
    """Extract layer name from task ID prefix."""
    layer_mapping = {
        'D-': 'Domain',
        'A-': 'Application',
        'I-': 'Infrastructure',
        'P-': 'Presentation',
        'C-': 'CI/CD',
        'DOC-': 'Documentation'
    }
    
    for prefix, layer in layer_mapping.items():
        if task_id.startswith(prefix):
            return layer
    return 'Unknown'


def parse_dependencies(deps_str: str) -> List[str]:
    """Parse dependencies string into list of task IDs."""
    if deps_str.strip().lower() in ['none', 'tbd', '']:
        return []
    
    # Extract task IDs from dependencies string
    deps = re.findall(r'[A-Z]+-\d+', deps_str)
    return deps


def parse_todo_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse TODO.md file and extract task information."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tasks = []
    
    # Pattern to match task sections
    task_pattern = r'###\s+([A-Z]+-\d+):\s+(.+?)\n(.*?)(?=###|---|\Z)'
    
    matches = re.finditer(task_pattern, content, re.DOTALL)
    
    for match in matches:
        task_id = match.group(1)
        title = match.group(2).strip()
        task_content = match.group(3)
        
        # Extract metadata from task content
        priority = ''
        estimate = 0
        description = ''
        dependencies = []
        
        # Parse each line in the task content
        lines = task_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- **Priority**:'):
                priority = line.split(':', 1)[1].strip()
            elif line.startswith('- **Estimate**:'):
                estimate_str = line.split(':', 1)[1].strip()
                estimate = parse_estimate(estimate_str)
            elif line.startswith('- **Dependencies**:'):
                deps_str = line.split(':', 1)[1].strip()
                dependencies = parse_dependencies(deps_str)
            elif line.startswith('- **Description**:'):
                description = line.split(':', 1)[1].strip()
        
        # Create task object
        task = {
            'id': task_id,
            'title': title,
            'layer': extract_layer_from_id(task_id),
            'priority': priority,
            'estimate': estimate,
            'description': description,
            'dependencies': dependencies,
            'suggested-milestone': map_priority_to_milestone(priority)
        }
        
        tasks.append(task)
    
    return tasks


def main():
    """Main function to parse TODO.md and emit JSON."""
    
    # Path to the TODO.md file
    todo_file = Path('docs/project/TODO.md')
    
    if not todo_file.exists():
        print(f"Error: {todo_file} not found")
        return
    
    try:
        # Parse the TODO file
        tasks = parse_todo_file(str(todo_file))
        
        # Output as JSON
        output = {
            'tasks': tasks,
            'total_tasks': len(tasks),
            'total_estimated_days': sum(task['estimate'] for task in tasks)
        }
        
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error parsing TODO.md: {e}")


if __name__ == '__main__':
    main()
