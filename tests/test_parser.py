#!/usr/bin/env python3
"""
Simple test script to verify the JSON output format and priority mapping.
"""

import json
import subprocess

def test_parser():
    """Test the parse_todo.py script output format."""
    
    # Run the parser and capture output
    result = subprocess.run(['python', 'parse_todo.py'], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running parser:", result.stderr)
        return False
    
    # Parse the JSON output
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print("Invalid JSON output:", e)
        return False
    
    # Verify structure
    if 'tasks' not in data:
        print("Missing 'tasks' key in output")
        return False
    
    # Check first task has all required fields
    if not data['tasks']:
        print("No tasks found")
        return False
    
    first_task = data['tasks'][0]
    required_fields = ['id', 'title', 'layer', 'priority', 'estimate', 'description', 'dependencies', 'suggested-milestone']
    
    for field in required_fields:
        if field not in first_task:
            print(f"Missing required field: {field}")
            return False
    
    # Verify priority mapping
    priority_samples = {
        'Critical': 'P1',
        'High': 'P2', 
        'Medium': 'P3',
        'Low': 'P4'
    }
    
    for task in data['tasks']:
        priority = task['priority']
        milestone = task['suggested-milestone']
        
        if priority in priority_samples:
            if milestone != priority_samples[priority]:
                print(f"Priority mapping error: {priority} -> {milestone}, expected {priority_samples[priority]}")
                return False
    
    print("✓ All tests passed!")
    print(f"✓ Parsed {len(data['tasks'])} tasks")
    print(f"✓ Total estimated days: {data.get('total_estimated_days', 'N/A')}")
    
    # Show priority distribution
    priority_counts = {}
    for task in data['tasks']:
        milestone = task['suggested-milestone']
        priority_counts[milestone] = priority_counts.get(milestone, 0) + 1
    
    print("✓ Priority distribution:")
    for milestone in sorted(priority_counts.keys()):
        print(f"  {milestone}: {priority_counts[milestone]} tasks")
    
    return True

if __name__ == '__main__':
    test_parser()
