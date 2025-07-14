#!/usr/bin/env python3
"""
Documentation Analysis Script
Analyzes documentation for broken links and orphaned files
"""

import os
import re
from collections import defaultdict


def find_markdown_files(docs_dir):
    """Find all markdown files in the documentation directory"""
    markdown_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))
    return markdown_files


def extract_links_from_file(file_path):
    """Extract all markdown links from a file"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Pattern to match markdown links [text](url)
        link_pattern = r"\[([^\]]*)\]\(([^)]+)\)"
        links = re.findall(link_pattern, content)

        return links
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def resolve_relative_path(current_file, link_path):
    """Resolve a relative path based on the current file location"""
    current_dir = os.path.dirname(current_file)

    # Remove URL fragments and query parameters
    clean_path = link_path.split("#")[0].split("?")[0]

    # Skip external URLs
    if clean_path.startswith(("http://", "https://", "mailto:", "tel:")):
        return None

    # Handle relative paths
    if clean_path.startswith("./"):
        clean_path = clean_path[2:]
        resolved_path = os.path.normpath(os.path.join(current_dir, clean_path))
    elif clean_path.startswith("../"):
        # Handle relative paths with ..
        resolved_path = os.path.normpath(os.path.join(current_dir, clean_path))
    else:
        # Handle paths that don't start with ./ or ../
        resolved_path = os.path.normpath(os.path.join(current_dir, clean_path))

    return resolved_path


def analyze_documentation(docs_dir):
    """Analyze documentation for broken links and orphaned files"""
    docs_dir = os.path.abspath(docs_dir)
    markdown_files = find_markdown_files(docs_dir)

    # Track all links and their sources
    all_links = defaultdict(list)  # target -> [source files]
    file_links = {}  # file -> [links]
    broken_links = []
    orphaned_files = set(markdown_files)

    print(f"Found {len(markdown_files)} markdown files")

    # Extract links from all files
    for md_file in markdown_files:
        links = extract_links_from_file(md_file)
        file_links[md_file] = links

        for link_text, link_url in links:
            # Resolve relative path
            resolved_path = resolve_relative_path(md_file, link_url)

            if resolved_path:
                # Check if the target exists
                target_file = resolved_path

                # If it's a directory, check for README.md or index.md
                if os.path.isdir(target_file):
                    for index_file in ["README.md", "index.md"]:
                        potential_target = os.path.join(target_file, index_file)
                        if os.path.exists(potential_target):
                            target_file = potential_target
                            break

                # Add to all_links tracking
                all_links[target_file].append(md_file)

                # Check if target exists
                if not os.path.exists(target_file):
                    broken_links.append(
                        {
                            "source": md_file,
                            "target": target_file,
                            "link_text": link_text,
                            "link_url": link_url,
                        }
                    )

    # Find orphaned files (files not referenced by any other file)
    for md_file in markdown_files:
        if md_file in all_links:
            orphaned_files.discard(md_file)
        # Index files are not orphaned by default
        if md_file.endswith(("index.md", "README.md")):
            orphaned_files.discard(md_file)

    return {
        "total_files": len(markdown_files),
        "broken_links": broken_links,
        "orphaned_files": list(orphaned_files),
        "file_links": file_links,
        "all_links": dict(all_links),
    }


def generate_report(analysis_result, docs_dir):
    """Generate a comprehensive report"""
    report = []

    report.append("# Documentation Analysis Report")
    report.append("=" * 50)
    report.append(f"Total markdown files: {analysis_result['total_files']}")
    report.append(f"Broken links found: {len(analysis_result['broken_links'])}")
    report.append(f"Orphaned files found: {len(analysis_result['orphaned_files'])}")
    report.append("")

    # Critical Issues Section
    report.append("## CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION")
    report.append("=" * 50)

    if analysis_result["broken_links"]:
        report.append("### 1. BROKEN LINKS")
        report.append("These links point to non-existent files:")
        report.append("")

        # Group by source file
        broken_by_source = defaultdict(list)
        for broken in analysis_result["broken_links"]:
            source = broken["source"].replace(docs_dir, "").lstrip("/")
            broken_by_source[source].append(broken)

        for source_file, broken_links in broken_by_source.items():
            report.append(f"**{source_file}:**")
            for broken in broken_links:
                target = broken["target"].replace(docs_dir, "").lstrip("/")
                report.append(
                    f"  - [{broken['link_text']}]({broken['link_url']}) -> {target}"
                )
            report.append("")

    # Orphaned Files Section
    if analysis_result["orphaned_files"]:
        report.append("### 2. ORPHANED FILES")
        report.append("These files are not referenced by any other documentation:")
        report.append("")

        for orphaned in sorted(analysis_result["orphaned_files"]):
            relative_path = orphaned.replace(docs_dir, "").lstrip("/")
            report.append(f"- {relative_path}")
        report.append("")

    # Navigation Structure Issues
    report.append("### 3. NAVIGATION STRUCTURE ISSUES")
    report.append("")

    # Check for missing key files
    key_files = [
        "index.md",
        "getting-started/README.md",
        "user-guides/README.md",
        "developer-guides/README.md",
        "reference/README.md",
        "deployment/README.md",
        "examples/README.md",
    ]

    missing_key_files = []
    for key_file in key_files:
        full_path = os.path.join(docs_dir, key_file)
        if not os.path.exists(full_path):
            missing_key_files.append(key_file)

    if missing_key_files:
        report.append("**Missing Key Navigation Files:**")
        for missing in missing_key_files:
            report.append(f"- {missing}")
        report.append("")

    # Link Patterns Analysis
    report.append("### 4. LINK PATTERNS ANALYSIS")
    report.append("")

    # Count different types of broken links
    missing_readme_count = 0
    missing_regular_count = 0
    directory_link_count = 0

    for broken in analysis_result["broken_links"]:
        target = broken["target"]
        if target.endswith("README.md"):
            missing_readme_count += 1
        elif os.path.isdir(os.path.dirname(target)) and not os.path.exists(target):
            missing_regular_count += 1
        elif broken["link_url"].endswith("/"):
            directory_link_count += 1

    report.append(f"- Missing README.md files: {missing_readme_count}")
    report.append(f"- Missing regular files: {missing_regular_count}")
    report.append(f"- Directory links without index: {directory_link_count}")
    report.append("")

    # Priority Recommendations
    report.append("## PRIORITY RECOMMENDATIONS")
    report.append("=" * 50)

    priorities = []

    if analysis_result["broken_links"]:
        priorities.append("1. **FIX BROKEN LINKS** - Immediate navigation failures")

    if missing_key_files:
        priorities.append(
            "2. **CREATE MISSING README FILES** - Critical navigation structure"
        )

    if analysis_result["orphaned_files"]:
        priorities.append("3. **INTEGRATE ORPHANED FILES** - Content accessibility")

    if priorities:
        for priority in priorities:
            report.append(priority)
        report.append("")

    report.append("## DETAILED ANALYSIS")
    report.append("=" * 50)

    # Most problematic files
    file_issues = defaultdict(int)
    for broken in analysis_result["broken_links"]:
        source = broken["source"]
        file_issues[source] += 1

    if file_issues:
        report.append("### Files with Most Issues:")
        sorted_issues = sorted(file_issues.items(), key=lambda x: x[1], reverse=True)
        for file_path, issue_count in sorted_issues[:10]:
            relative_path = file_path.replace(docs_dir, "").lstrip("/")
            report.append(f"- {relative_path}: {issue_count} broken links")
        report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    docs_dir = "/mnt/c/Users/andre/Pynomaly/docs"

    print("Starting documentation analysis...")
    analysis = analyze_documentation(docs_dir)

    print("Generating report...")
    report = generate_report(analysis, docs_dir)

    print(report)

    # Also save to file
    with open("/mnt/c/Users/andre/Pynomaly/docs_analysis_report.txt", "w") as f:
        f.write(report)

    print("\nFull report saved to docs_analysis_report.txt")
