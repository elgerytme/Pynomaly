#!/usr/bin/env python3
"""
Automated stash processing script for Step 8: Process 78+ stashes methodically
"""

import subprocess
import sys
import json
import os
from pathlib import Path
import re
from datetime import datetime

class StashProcessor:
    def __init__(self):
        self.audit_file = Path("docs/stash_audit_2025-07-09.md")
        self.results = []
        self.processed_count = 0
        
    def get_stash_list(self):
        """Get list of all stashes"""
        try:
            result = subprocess.run(
                ["git", "stash", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            stashes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # Parse stash entry: stash@{0}: On main: Description
                    match = re.match(r'(stash@\{(\d+)\}):\s*(.+)', line)
                    if match:
                        stash_ref = match.group(1)
                        index = match.group(2)
                        description = match.group(3)
                        stashes.append({
                            'ref': stash_ref,
                            'index': int(index),
                            'description': description
                        })
            return stashes
        except subprocess.CalledProcessError as e:
            print(f"Error getting stash list: {e}")
            return []
    
    def get_stash_hash(self, stash_ref):
        """Get short hash for stash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", stash_ref],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def create_stash_branch(self, index, hash_val):
        """Create branch for stash processing"""
        branch_name = f"stash/{index}-{hash_val}"
        try:
            # Create new branch from current HEAD
            subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
            return branch_name
        except subprocess.CalledProcessError as e:
            print(f"Error creating branch {branch_name}: {e}")
            return None
    
    def apply_stash(self, stash_ref):
        """Apply stash to current branch"""
        try:
            # First try to apply the stash
            result = subprocess.run(["git", "stash", "apply", stash_ref], capture_output=True, text=True)
            if result.returncode == 0:
                return True, "Applied successfully"
            else:
                # Check if it's a conflict
                if "CONFLICT" in result.stderr or "conflict" in result.stderr:
                    # Try to resolve conflicts automatically by taking their version
                    subprocess.run(["git", "checkout", "--theirs", "."], capture_output=True)
                    subprocess.run(["git", "add", "."], capture_output=True)
                    return True, "Applied with conflict resolution"
                else:
                    # Try alternative approach using show and patch
                    return self.apply_stash_alternative(stash_ref)
        except subprocess.CalledProcessError as e:
            return False, f"Error applying stash {stash_ref}: {e}"
    
    def apply_stash_alternative(self, stash_ref):
        """Alternative method to apply stash using show and patch"""
        try:
            # Get the stash as a patch
            result = subprocess.run(["git", "stash", "show", "-p", stash_ref], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                # Try to apply the patch
                patch_result = subprocess.run(["git", "apply", "--ignore-whitespace", "--3way"], 
                                            input=result.stdout, capture_output=True, text=True)
                if patch_result.returncode == 0:
                    return True, "Applied using patch method"
                else:
                    return False, f"Patch apply failed: {patch_result.stderr}"
            else:
                return False, "Could not generate patch"
        except subprocess.CalledProcessError as e:
            return False, f"Alternative apply failed: {e}"
    
    def run_tests(self):
        """Run tests to check if stash is valid"""
        try:
            # Try pytest first
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode == 0:
                return True, "Tests passed"
            else:
                return False, f"Tests failed: {result.stdout[-500:]}"  # Last 500 chars
        except subprocess.TimeoutExpired:
            return False, "Tests timed out"
        except FileNotFoundError:
            # Try with python -m pytest if pytest not found
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", "tests/", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return True, "Tests passed"
                else:
                    return False, f"Tests failed: {result.stdout[-500:]}"
            except:
                return False, "Could not run tests"
    
    def check_relevance(self, stash_ref):
        """Check if stash contains relevant changes"""
        try:
            result = subprocess.run(
                ["git", "stash", "show", "--stat", stash_ref],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if stash has meaningful changes (not just whitespace, etc.)
            output = result.stdout
            if not output.strip():
                return False, "No file changes"
            
            # Check for relevant file types
            relevant_patterns = [
                r'\.py\s',  # Python files
                r'\.md\s',  # Markdown files
                r'\.yml\s', r'\.yaml\s',  # YAML files
                r'\.json\s',  # JSON files
                r'\.txt\s',  # Text files
                r'\.toml\s',  # TOML files
            ]
            
            for pattern in relevant_patterns:
                if re.search(pattern, output):
                    return True, "Contains relevant files"
            
            return False, "No relevant files found"
            
        except subprocess.CalledProcessError:
            return False, "Could not analyze stash"
    
    def commit_changes(self, stash_ref, description):
        """Commit the applied stash changes"""
        try:
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            commit_msg = f"Apply {stash_ref}: {description}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error committing changes: {e}")
            return False
    
    def cleanup_branch(self, branch_name):
        """Clean up branch after processing"""
        try:
            subprocess.run(["git", "checkout", "main"], check=True, capture_output=True)
            subprocess.run(["git", "branch", "-D", branch_name], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cleaning up branch {branch_name}: {e}")
    
    def update_audit_file(self, index, hash_val, status, notes=""):
        """Update audit markdown file with processing results"""
        content = f"| {index} | {hash_val} | {status} | {notes[:100]}... |\n"
        
        # Read current content
        try:
            with open(self.audit_file, 'r') as f:
                current_content = f.read()
        except FileNotFoundError:
            current_content = ""
        
        # Add new entry
        if "| Index | Hash    | Status  |" in current_content:
            # Insert after header
            lines = current_content.split('\n')
            header_index = -1
            for i, line in enumerate(lines):
                if line.startswith('|-------|'):
                    header_index = i
                    break
            
            if header_index >= 0:
                lines.insert(header_index + 1, content.rstrip())
                new_content = '\n'.join(lines)
            else:
                new_content = current_content + content
        else:
            new_content = current_content + content
        
        # Write back
        with open(self.audit_file, 'w') as f:
            f.write(new_content)
    
    def process_stash(self, stash_info):
        """Process a single stash"""
        stash_ref = stash_info['ref']
        index = stash_info['index']
        description = stash_info['description']
        
        print(f"\n=== Processing {stash_ref}: {description} ===")
        
        # Get hash
        hash_val = self.get_stash_hash(stash_ref)
        
        # Check relevance first
        is_relevant, relevance_note = self.check_relevance(stash_ref)
        if not is_relevant:
            print(f"Skipping {stash_ref}: {relevance_note}")
            self.update_audit_file(index, hash_val, "SKIPPED", relevance_note)
            return
        
        # Create branch
        branch_name = self.create_stash_branch(index, hash_val)
        if not branch_name:
            self.update_audit_file(index, hash_val, "FAILED", "Could not create branch")
            return
        
        try:
            # Apply stash
            applied, apply_notes = self.apply_stash(stash_ref)
            if not applied:
                self.update_audit_file(index, hash_val, "FAILED", f"Could not apply stash: {apply_notes}")
                self.cleanup_branch(branch_name)
                return
            
            # Run tests
            tests_passed, test_notes = self.run_tests()
            
            if tests_passed and is_relevant:
                # Commit changes
                if self.commit_changes(stash_ref, description):
                    print(f"✓ {stash_ref} committed successfully")
                    self.update_audit_file(index, hash_val, "COMMITTED", f"Tests passed. {test_notes}")
                    # Keep branch for potential PR
                else:
                    print(f"✗ {stash_ref} failed to commit")
                    self.update_audit_file(index, hash_val, "FAILED", "Could not commit changes")
                    self.cleanup_branch(branch_name)
            else:
                print(f"✗ {stash_ref} tests failed or not relevant")
                self.update_audit_file(index, hash_val, "DROPPED", f"Tests failed or not relevant: {test_notes}")
                self.cleanup_branch(branch_name)
        
        except Exception as e:
            print(f"Error processing {stash_ref}: {e}")
            self.update_audit_file(index, hash_val, "ERROR", str(e))
            self.cleanup_branch(branch_name)
        
        self.processed_count += 1
    
    def process_all_stashes(self):
        """Process all stashes systematically"""
        print("Starting automated stash processing...")
        
        # Get all stashes
        stashes = self.get_stash_list()
        if not stashes:
            print("No stashes found")
            return
        
        print(f"Found {len(stashes)} stashes to process")
        
        # Process each stash
        for stash_info in stashes:
            self.process_stash(stash_info)
        
        print(f"\n=== Processing Complete ===")
        print(f"Processed {self.processed_count} stashes")
        print(f"Audit file: {self.audit_file}")

def main():
    """Main entry point"""
    processor = StashProcessor()
    processor.process_all_stashes()

if __name__ == "__main__":
    main()
