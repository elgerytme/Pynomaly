#!/usr/bin/env python3
"""
Documentation Date Validation Script

Validates that dates in documentation files use the correct local timezone
and are not off by days, months, or years.

Usage:
    python scripts/automation/validate_dates.py [--fix]

Options:
    --fix: Automatically fix incorrect dates to current local date
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


class DateValidator:
    """Validates and fixes dates in documentation files."""

    def __init__(self, fix_mode: bool = False):
        self.fix_mode = fix_mode
        self.local_tz = datetime.now().astimezone().tzinfo
        self.current_date = datetime.now(self.local_tz)
        self.errors_found = []

        # Date patterns to match various formats
        self.date_patterns = [
            # "July 11, 2025", "Jul 11, 2025"
            r"(\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})",
            # "2025-07-11", "11/07/2025", "07-11-2025"
            r"(\b\d{4}-\d{1,2}-\d{1,2}\b)",
            r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b)",
            # Markdown date formats
            r"(\*\*Last Updated\*\*:\s*[^\n\r]+)",
            r"(\*\*Date\*\*:\s*[^\n\r]+)",
            r"(\*\*Created\*\*:\s*[^\n\r]+)",
            # Comments with dates
            r"(#\s*Last updated:\s*[^\n\r]+)",
            r"(#\s*Date:\s*[^\n\r]+)",
            # YAML frontmatter dates
            r"(date:\s*[^\n\r]+)",
            r"(updated:\s*[^\n\r]+)",
            r"(created:\s*[^\n\r]+)",
        ]

        # Files to check
        self.documentation_files = [
            "README.md",
            "TODO.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "docs/**/*.md",
            "scripts/automation/README.md",
            ".github/workflows/*.yml",
            ".github/workflows/*.yaml",
            "**/*.md",
        ]

    def get_files_to_check(self) -> list[Path]:
        """Get list of files to check for date validation."""
        files = []
        root_path = Path(".")

        for pattern in self.documentation_files:
            if "**" in pattern:
                # Recursive glob
                files.extend(root_path.glob(pattern))
            else:
                # Direct glob
                files.extend(root_path.glob(pattern))

        # Remove duplicates and filter to existing files
        unique_files = list(set(files))
        existing_files = [f for f in unique_files if f.exists() and f.is_file()]

        return existing_files

    def extract_date_from_text(self, text: str) -> datetime | None:
        """Extract date from text and convert to datetime object."""
        text = text.strip()

        # Try parsing various date formats
        date_formats = [
            "%B %d, %Y",  # "July 11, 2025"
            "%b %d, %Y",  # "Jul 11, 2025"
            "%Y-%m-%d",  # "2025-07-11"
            "%m/%d/%Y",  # "07/11/2025"
            "%d/%m/%Y",  # "11/07/2025"
            "%m-%d-%Y",  # "07-11-2025"
            "%d-%m-%Y",  # "11-07-2025"
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue

        return None

    def is_date_reasonable(self, date: datetime) -> tuple[bool, str]:
        """Check if date is reasonable (not too far off from current date)."""
        if not date:
            return True, ""

        current = self.current_date.replace(tzinfo=None)
        diff = abs((date - current).days)

        # Allow dates up to 30 days in the future or past
        if diff > 30:
            if date > current:
                return False, f"Date is {diff} days in the future"
            else:
                return False, f"Date is {diff} days in the past"

        return True, ""

    def get_correct_date_format(self, original_text: str) -> str:
        """Get the correctly formatted current date matching the original format."""
        # Detect the format of the original text and return current date in same format
        if re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            original_text,
        ):
            return self.current_date.strftime("%B %d, %Y")
        elif re.search(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
            original_text,
        ):
            return self.current_date.strftime("%b %d, %Y")
        elif re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", original_text):
            return self.current_date.strftime("%Y-%m-%d")
        elif re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", original_text):
            return self.current_date.strftime("%m/%d/%Y")
        else:
            # Default to full month format
            return self.current_date.strftime("%B %d, %Y")

    def validate_file(self, file_path: Path) -> list[tuple[int, str, str]]:
        """Validate dates in a single file. Returns list of (line_num, line_content, issue)."""
        issues = []

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError):
            return issues

        for line_num, line in enumerate(lines, 1):
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group(1)

                    # Extract just the date part
                    date_text = re.sub(r"^\*\*[^:]+\*\*:\s*", "", matched_text)
                    date_text = re.sub(r"^[^:]+:\s*", "", date_text)
                    date_text = date_text.strip()

                    # Try to parse the date
                    parsed_date = self.extract_date_from_text(date_text)
                    if parsed_date:
                        is_reasonable, reason = self.is_date_reasonable(parsed_date)
                        if not is_reasonable:
                            issues.append((line_num, line.strip(), reason))

        return issues

    def fix_file(self, file_path: Path, issues: list[tuple[int, str, str]]) -> bool:
        """Fix dates in a file. Returns True if file was modified."""
        if not issues or not self.fix_mode:
            return False

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            return False

        modified = False

        # Fix each pattern
        for pattern in self.date_patterns:

            def replace_date(match):
                nonlocal modified
                original = match.group(1)

                # Extract the date part
                date_part = re.sub(r"^\*\*[^:]+\*\*:\s*", "", original)
                date_part = re.sub(r"^[^:]+:\s*", "", date_part)
                date_part = date_part.strip()

                parsed_date = self.extract_date_from_text(date_part)
                if parsed_date:
                    is_reasonable, _ = self.is_date_reasonable(parsed_date)
                    if not is_reasonable:
                        # Replace with current date in same format
                        new_date = self.get_correct_date_format(original)

                        # Preserve the original structure
                        if original.startswith("**"):
                            prefix = re.match(r"^\*\*[^:]+\*\*:\s*", original)
                            if prefix:
                                result = prefix.group(0) + new_date
                            else:
                                result = new_date
                        elif ":" in original:
                            prefix = re.match(r"^[^:]+:\s*", original)
                            if prefix:
                                result = prefix.group(0) + new_date
                            else:
                                result = new_date
                        else:
                            result = new_date

                        modified = True
                        return result

                return original

            content = re.sub(pattern, replace_date, content, flags=re.IGNORECASE)

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return modified

    def run(self) -> bool:
        """Run date validation on all documentation files."""
        print("🔍 Scanning documentation files for date issues...")
        print(
            f"📅 Current local time: {self.current_date.strftime('%B %d, %Y %H:%M %Z')}"
        )

        files_to_check = self.get_files_to_check()
        print(f"📁 Found {len(files_to_check)} documentation files to check")

        total_issues = 0
        files_with_issues = 0
        files_fixed = 0

        for file_path in files_to_check:
            issues = self.validate_file(file_path)

            if issues:
                files_with_issues += 1
                total_issues += len(issues)

                print(f"\n❌ {file_path}:")
                for line_num, line_content, issue in issues:
                    print(f"   Line {line_num}: {issue}")
                    print(f"   Content: {line_content}")

                # Try to fix if in fix mode
                if self.fix_mode:
                    if self.fix_file(file_path, issues):
                        files_fixed += 1
                        print(f"   ✅ Fixed dates in {file_path}")

        # Summary
        print("\n📊 Summary:")
        print(f"   Files checked: {len(files_to_check)}")
        print(f"   Files with date issues: {files_with_issues}")
        print(f"   Total date issues found: {total_issues}")

        if self.fix_mode:
            print(f"   Files fixed: {files_fixed}")

        if total_issues > 0:
            if not self.fix_mode:
                print("\n💡 Run with --fix to automatically correct these dates")
            return False
        else:
            print("\n✅ All documentation dates are correct!")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate dates in documentation files"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix incorrect dates"
    )
    args = parser.parse_args()

    try:
        validator = DateValidator(fix_mode=args.fix)
        success = validator.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
