#!/usr/bin/env python3
"""
Documentation Link Checker

Analyzes all documentation files for broken internal links and cross-references.
Checks for:
1. Links to non-existent files
2. References to moved or renamed files
3. Internal cross-references that are broken
4. Links to examples or guides that don't exist
5. Documentation that references outdated information
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse


@dataclass
class BrokenLink:
    """Represents a broken link found in documentation"""

    source_file: str
    link_text: str
    target_path: str
    line_number: int
    issue_type: str
    context: str
    suggested_fix: Optional[str] = None


@dataclass
class LinkAnalysisResult:
    """Results of link analysis"""

    broken_links: list[BrokenLink]
    total_links_checked: int
    files_analyzed: int
    critical_files_missing: list[str]
    recommendations: list[str]


class DocumentationLinkChecker:
    """Analyzes documentation files for broken links and cross-references"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.examples_dir = project_root / "examples"
        self.scripts_dir = project_root / "scripts"
        self.src_dir = project_root / "src"

        # Pattern to match markdown links
        self.markdown_link_pattern = re.compile(
            r"\[([^\]]+)\]\(([^)]+)\)", re.MULTILINE
        )

        # Pattern to match file references
        self.file_ref_pattern = re.compile(
            r'(?:see|refer to|check|view|read|documentation at|guide at|found in|located in|available at)\s+[`"]?([^`"\s]+\.[a-zA-Z0-9]+)[`"]?',
            re.IGNORECASE,
        )

        # Pattern to match code references
        self.code_ref_pattern = re.compile(r"`([^`]+\.[a-zA-Z0-9]+)`", re.MULTILINE)

        # Common documentation extensions
        self.doc_extensions = {".md", ".rst", ".txt", ".html", ".yaml", ".yml", ".json"}

        # Files that should exist based on README references
        self.critical_files = {
            "docs/WEB_API_SETUP_GUIDE.md",
            "docs/API_QUICK_REFERENCE.md",
            "docs/developer-guides/contributing/CONTRIBUTING.md",
            "CHANGELOG.md",
            "TODO.md",
            "LICENSE",
        }

    def analyze_documentation(self) -> LinkAnalysisResult:
        """Analyze all documentation files for broken links"""
        broken_links = []
        total_links = 0
        files_analyzed = 0

        # Get all documentation files
        doc_files = self._get_documentation_files()

        for file_path in doc_files:
            try:
                file_broken_links, file_link_count = self._analyze_file(file_path)
                broken_links.extend(file_broken_links)
                total_links += file_link_count
                files_analyzed += 1

                if file_link_count > 0:
                    print(
                        f"Analyzed {file_path.relative_to(self.project_root)}: {file_link_count} links, {len(file_broken_links)} broken"
                    )

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        # Check for critical missing files
        critical_missing = self._check_critical_files()

        # Generate recommendations
        recommendations = self._generate_recommendations(broken_links, critical_missing)

        return LinkAnalysisResult(
            broken_links=broken_links,
            total_links_checked=total_links,
            files_analyzed=files_analyzed,
            critical_files_missing=critical_missing,
            recommendations=recommendations,
        )

    def _get_documentation_files(self) -> list[Path]:
        """Get all documentation files to analyze"""
        files = []

        # Documentation directories to check
        dirs_to_check = [
            self.docs_dir,
            self.project_root,  # Root level files like README.md
            self.examples_dir,
            (
                self.scripts_dir / "documentation"
                if (self.scripts_dir / "documentation").exists()
                else None
            ),
        ]

        for directory in dirs_to_check:
            if directory and directory.exists():
                for ext in self.doc_extensions:
                    files.extend(directory.rglob(f"*{ext}"))

        # Add specific root files
        root_files = ["README.md", "CHANGELOG.md", "TODO.md", "CONTRIBUTING.md"]
        for filename in root_files:
            file_path = self.project_root / filename
            if file_path.exists():
                files.append(file_path)

        return list(set(files))  # Remove duplicates

    def _analyze_file(self, file_path: Path) -> tuple[list[BrokenLink], int]:
        """Analyze a single file for broken links"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Skip binary files
            return [], 0

        broken_links = []
        total_links = 0

        # Analyze markdown links
        markdown_broken, markdown_total = self._check_markdown_links(file_path, content)
        broken_links.extend(markdown_broken)
        total_links += markdown_total

        # Analyze file references
        ref_broken, ref_total = self._check_file_references(file_path, content)
        broken_links.extend(ref_broken)
        total_links += ref_total

        # Analyze code references
        code_broken, code_total = self._check_code_references(file_path, content)
        broken_links.extend(code_broken)
        total_links += code_total

        return broken_links, total_links

    def _check_markdown_links(
        self, file_path: Path, content: str
    ) -> tuple[list[BrokenLink], int]:
        """Check markdown-style links [text](path)"""
        broken_links = []
        matches = list(self.markdown_link_pattern.finditer(content))

        for match in matches:
            link_text = match.group(1)
            link_target = match.group(2)
            line_number = content[: match.start()].count("\n") + 1
            context = self._get_context(content, match.start(), match.end())

            # Skip external URLs
            if self._is_external_url(link_target):
                continue

            # Check if target exists
            if not self._target_exists(file_path, link_target):
                suggested_fix = self._suggest_fix(link_target)
                broken_links.append(
                    BrokenLink(
                        source_file=str(file_path.relative_to(self.project_root)),
                        link_text=link_text,
                        target_path=link_target,
                        line_number=line_number,
                        issue_type="markdown_link",
                        context=context,
                        suggested_fix=suggested_fix,
                    )
                )

        return broken_links, len(matches)

    def _check_file_references(
        self, file_path: Path, content: str
    ) -> tuple[list[BrokenLink], int]:
        """Check file references in text"""
        broken_links = []
        matches = list(self.file_ref_pattern.finditer(content))

        for match in matches:
            target_path = match.group(1)
            line_number = content[: match.start()].count("\n") + 1
            context = self._get_context(content, match.start(), match.end())

            # Skip if it looks like a URL or code
            if self._is_external_url(target_path) or "://" in target_path:
                continue

            # Check if target exists
            if not self._target_exists(file_path, target_path):
                suggested_fix = self._suggest_fix(target_path)
                broken_links.append(
                    BrokenLink(
                        source_file=str(file_path.relative_to(self.project_root)),
                        link_text=match.group(0),
                        target_path=target_path,
                        line_number=line_number,
                        issue_type="file_reference",
                        context=context,
                        suggested_fix=suggested_fix,
                    )
                )

        return broken_links, len(matches)

    def _check_code_references(
        self, file_path: Path, content: str
    ) -> tuple[list[BrokenLink], int]:
        """Check code-style references `path/to/file.ext`"""
        broken_links = []
        matches = list(self.code_ref_pattern.finditer(content))

        for match in matches:
            target_path = match.group(1)
            line_number = content[: match.start()].count("\n") + 1
            context = self._get_context(content, match.start(), match.end())

            # Skip if it doesn't look like a file path
            if not self._looks_like_file_path(target_path):
                continue

            # Check if target exists
            if not self._target_exists(file_path, target_path):
                suggested_fix = self._suggest_fix(target_path)
                broken_links.append(
                    BrokenLink(
                        source_file=str(file_path.relative_to(self.project_root)),
                        link_text=match.group(0),
                        target_path=target_path,
                        line_number=line_number,
                        issue_type="code_reference",
                        context=context,
                        suggested_fix=suggested_fix,
                    )
                )

        return broken_links, len(matches)

    def _target_exists(self, source_file: Path, target_path: str) -> bool:
        """Check if a target path exists"""
        # Clean up the target path
        target_path = unquote(target_path)
        target_path = target_path.split("#")[0]  # Remove fragment
        target_path = target_path.split("?")[0]  # Remove query

        if not target_path:
            return True

        # Try different path resolutions
        possible_paths = [
            # Absolute from project root
            self.project_root / target_path,
            # Relative to source file
            source_file.parent / target_path,
            # Relative to docs directory
            self.docs_dir / target_path,
            # Direct path if already absolute
            Path(target_path) if Path(target_path).is_absolute() else None,
        ]

        for path in possible_paths:
            if path and path.exists():
                return True

        return False

    def _is_external_url(self, url: str) -> bool:
        """Check if URL is external"""
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)

    def _looks_like_file_path(self, path: str) -> bool:
        """Check if string looks like a file path"""
        # Has file extension
        if "." in path and path.split(".")[-1] in [
            "md",
            "py",
            "yml",
            "yaml",
            "json",
            "txt",
            "html",
            "rst",
        ]:
            return True

        # Has path separators
        if "/" in path or "\\" in path:
            return True

        return False

    def _get_context(self, content: str, start: int, end: int) -> str:
        """Get surrounding context for a match"""
        lines = content.split("\n")
        line_num = content[:start].count("\n")

        # Get surrounding lines
        context_lines = []
        for i in range(max(0, line_num - 1), min(len(lines), line_num + 2)):
            prefix = ">>> " if i == line_num else "    "
            context_lines.append(f"{prefix}{lines[i]}")

        return "\n".join(context_lines)

    def _suggest_fix(self, target_path: str) -> Optional[str]:
        """Suggest a fix for a broken link"""
        # Look for similar files
        target_name = Path(target_path).name

        # Search for files with similar names
        similar_files = []
        for file_path in self._get_documentation_files():
            if file_path.name.lower() == target_name.lower():
                similar_files.append(str(file_path.relative_to(self.project_root)))

        if similar_files:
            return f"Similar files found: {', '.join(similar_files)}"

        # Check if it's a moved file
        if target_path.startswith("docs/"):
            # Check if file exists elsewhere
            filename = Path(target_path).name
            for file_path in self._get_documentation_files():
                if file_path.name == filename:
                    return f"File may have moved to: {file_path.relative_to(self.project_root)}"

        return None

    def _check_critical_files(self) -> list[str]:
        """Check for critical files that should exist"""
        missing = []

        for file_path in self.critical_files:
            if not (self.project_root / file_path).exists():
                missing.append(file_path)

        return missing

    def _generate_recommendations(
        self, broken_links: list[BrokenLink], critical_missing: list[str]
    ) -> list[str]:
        """Generate recommendations for fixing issues"""
        recommendations = []

        if critical_missing:
            recommendations.append(
                f"Create missing critical files: {', '.join(critical_missing)}"
            )

        # Group broken links by type
        by_type = {}
        for link in broken_links:
            if link.issue_type not in by_type:
                by_type[link.issue_type] = []
            by_type[link.issue_type].append(link)

        for issue_type, links in by_type.items():
            recommendations.append(
                f"Fix {len(links)} broken {issue_type.replace('_', ' ')} links"
            )

        # Specific recommendations
        if broken_links:
            recommendations.extend(
                [
                    "Update links to use correct relative paths from the source file",
                    "Consider using absolute paths from project root where appropriate",
                    "Verify that referenced files exist at the expected locations",
                    "Update documentation after moving or renaming files",
                ]
            )

        return recommendations


def main():
    """Run the documentation link checker"""
    project_root = Path(__file__).parent.parent.parent
    checker = DocumentationLinkChecker(project_root)

    print("🔍 Analyzing documentation links...")
    result = checker.analyze_documentation()

    print("\n📊 Analysis Results:")
    print(f"   Files analyzed: {result.files_analyzed}")
    print(f"   Total links checked: {result.total_links_checked}")
    print(f"   Broken links found: {len(result.broken_links)}")
    print(f"   Critical files missing: {len(result.critical_files_missing)}")

    if result.critical_files_missing:
        print("\n❌ Missing Critical Files:")
        for file in result.critical_files_missing:
            print(f"   - {file}")

    if result.broken_links:
        print("\n🔗 Broken Links Found:")

        # Group by source file
        by_file = {}
        for link in result.broken_links:
            if link.source_file not in by_file:
                by_file[link.source_file] = []
            by_file[link.source_file].append(link)

        for source_file, links in by_file.items():
            print(f"\n📄 {source_file}:")
            for link in links:
                print(f"   Line {link.line_number}: {link.target_path}")
                print(f"   Type: {link.issue_type}")
                print(
                    f"   Context: {link.context.split(chr(10))[1] if chr(10) in link.context else link.context}"
                )
                if link.suggested_fix:
                    print(f"   Suggestion: {link.suggested_fix}")
                print()

    if result.recommendations:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")

    # Save detailed results
    results_file = project_root / "reports" / "documentation_link_analysis.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "files_analyzed": result.files_analyzed,
                    "total_links_checked": result.total_links_checked,
                    "broken_links_count": len(result.broken_links),
                    "critical_files_missing_count": len(result.critical_files_missing),
                },
                "critical_files_missing": result.critical_files_missing,
                "broken_links": [
                    {
                        "source_file": link.source_file,
                        "link_text": link.link_text,
                        "target_path": link.target_path,
                        "line_number": link.line_number,
                        "issue_type": link.issue_type,
                        "context": link.context,
                        "suggested_fix": link.suggested_fix,
                    }
                    for link in result.broken_links
                ],
                "recommendations": result.recommendations,
            },
            indent=2,
        )

    print(f"\n💾 Detailed results saved to: {results_file}")

    # Return appropriate exit code
    if result.broken_links or result.critical_files_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
