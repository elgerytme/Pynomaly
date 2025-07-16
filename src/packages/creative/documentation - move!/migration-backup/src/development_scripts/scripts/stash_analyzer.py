#!/usr/bin/env python3
"""
Simplified stash analyzer for Step 8: Process 78+ stashes methodically
Analyzes stashes without applying them to avoid conflicts
"""

import re
import subprocess
from pathlib import Path


class StashAnalyzer:
    def __init__(self):
        self.audit_file = Path("docs/stash_audit_2025-07-09.md")
        self.results = []
        self.processed_count = 0

    def get_stash_list(self):
        """Get list of all stashes"""
        try:
            result = subprocess.run(
                ["git", "stash", "list"], capture_output=True, text=True, check=True
            )
            stashes = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    # Parse stash entry: stash@{0}: On main: Description
                    match = re.match(r"(stash@\{(\d+)\}):\s*(.+)", line)
                    if match:
                        stash_ref = match.group(1)
                        index = match.group(2)
                        description = match.group(3)
                        stashes.append(
                            {
                                "ref": stash_ref,
                                "index": int(index),
                                "description": description,
                            }
                        )
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
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"

    def analyze_stash_content(self, stash_ref):
        """Analyze stash content without applying it"""
        try:
            # Get stash diff stats
            result = subprocess.run(
                ["git", "stash", "show", "--stat", stash_ref],
                capture_output=True,
                text=True,
                check=True,
            )

            if not result.stdout.strip():
                return False, "No file changes", []

            # Parse file changes
            files_changed = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    filename = line.split("|")[0].strip()
                    files_changed.append(filename)

            # Check for relevant file types
            relevant_files = []
            relevant_patterns = [
                r"\.py$",  # Python files
                r"\.md$",  # Markdown files
                r"\.yml$",
                r"\.yaml$",  # YAML files
                r"\.json$",  # JSON files
                r"\.txt$",  # Text files
                r"\.toml$",  # TOML files
                r"\.ini$",  # INI files
            ]

            for filename in files_changed:
                for pattern in relevant_patterns:
                    if re.search(pattern, filename):
                        relevant_files.append(filename)
                        break

            if relevant_files:
                return (
                    True,
                    f"Contains {len(relevant_files)} relevant files",
                    relevant_files,
                )
            else:
                return False, "No relevant files found", files_changed

        except subprocess.CalledProcessError:
            return False, "Could not analyze stash", []

    def get_stash_summary(self, stash_ref):
        """Get summary of stash changes"""
        try:
            result = subprocess.run(
                ["git", "stash", "show", "--name-only", stash_ref],
                capture_output=True,
                text=True,
                check=True,
            )

            files = result.stdout.strip().split("\n") if result.stdout.strip() else []

            # Get insertion/deletion counts
            stat_result = subprocess.run(
                ["git", "stash", "show", "--stat", stash_ref],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the summary line (e.g., "5 files changed, 123 insertions(+), 45 deletions(-)")
            summary_lines = stat_result.stdout.strip().split("\n")
            summary = summary_lines[-1] if summary_lines else ""

            return files, summary

        except subprocess.CalledProcessError:
            return [], "Could not get summary"

    def update_audit_file(self, index, hash_val, status, notes="", files_changed=None):
        """Update audit markdown file with analysis results"""
        files_str = f" ({len(files_changed)} files)" if files_changed else ""
        content = f"| {index} | {hash_val} | {status} | {notes[:80]}...{files_str} |\n"

        # Read current content
        try:
            with open(self.audit_file) as f:
                current_content = f.read()
        except FileNotFoundError:
            current_content = ""

        # Add new entry
        if "|-------|---------|---------|-------|" in current_content:
            # Insert after header
            lines = current_content.split("\n")
            header_index = -1
            for i, line in enumerate(lines):
                if line.startswith("|-------|"):
                    header_index = i
                    break

            if header_index >= 0:
                lines.insert(header_index + 1, content.rstrip())
                new_content = "\n".join(lines)
            else:
                new_content = current_content + content
        else:
            new_content = current_content + content

        # Write back
        with open(self.audit_file, "w") as f:
            f.write(new_content)

    def analyze_stash(self, stash_info):
        """Analyze a single stash"""
        stash_ref = stash_info["ref"]
        index = stash_info["index"]
        description = stash_info["description"]

        print(f"\n=== Analyzing {stash_ref}: {description} ===")

        # Get hash
        hash_val = self.get_stash_hash(stash_ref)

        # Analyze content
        is_relevant, analysis_note, files_changed = self.analyze_stash_content(
            stash_ref
        )

        # Get summary
        files, summary = self.get_stash_summary(stash_ref)

        if is_relevant:
            print(f"✓ {stash_ref} contains relevant changes: {analysis_note}")
            print(f"  Summary: {summary}")
            print(
                f"  Files: {', '.join(files_changed[:5])}{'...' if len(files_changed) > 5 else ''}"
            )

            # Mark as candidate for manual review
            self.update_audit_file(
                index,
                hash_val,
                "CANDIDATE",
                f"Relevant changes found. {analysis_note}. Summary: {summary}",
                files_changed,
            )
        else:
            print(f"⚠ {stash_ref} skipped: {analysis_note}")
            self.update_audit_file(
                index, hash_val, "SKIPPED", analysis_note, files_changed
            )

        self.processed_count += 1

    def analyze_all_stashes(self):
        """Analyze all stashes systematically"""
        print("Starting stash analysis...")

        # Get all stashes
        stashes = self.get_stash_list()
        if not stashes:
            print("No stashes found")
            return

        print(f"Found {len(stashes)} stashes to analyze")

        # Analyze each stash
        for stash_info in stashes:
            self.analyze_stash(stash_info)

        print("\n=== Analysis Complete ===")
        print(f"Analyzed {self.processed_count} stashes")
        print(f"Audit file: {self.audit_file}")

        # Print summary
        try:
            with open(self.audit_file) as f:
                content = f.read()
                candidate_count = content.count("| CANDIDATE |")
                skipped_count = content.count("| SKIPPED |")
                print(f"  - {candidate_count} candidates for manual review")
                print(f"  - {skipped_count} skipped (not relevant)")
        except:
            pass


def main():
    """Main entry point"""
    analyzer = StashAnalyzer()
    analyzer.analyze_all_stashes()


if __name__ == "__main__":
    main()
