#!/usr/bin/env python3
"""
Quick Documentation Link Checker

Focused check for the most critical broken links in documentation.
"""

import os
import re
from pathlib import Path


def check_critical_links():
    """Check critical links that are referenced in main documentation"""

    project_root = Path(__file__).parent.parent.parent
    issues = []

    # Critical files mentioned in README.md
    critical_files = [
        "docs/WEB_API_SETUP_GUIDE.md",
        "docs/API_QUICK_REFERENCE.md",
        "docs/developer-guides/contributing/CONTRIBUTING.md",
        "CHANGELOG.md",
        "TODO.md",
    ]

    print("🔍 Checking critical files referenced in README.md...")

    for file_path in critical_files:
        full_path = project_root / file_path
        if not full_path.exists():
            # Check if file exists in different location
            filename = Path(file_path).name
            found_paths = []

            # Search in docs directory
            for found_file in project_root.rglob(filename):
                if found_file.is_file():
                    found_paths.append(str(found_file.relative_to(project_root)))

            if found_paths:
                issue = {
                    "type": "moved_file",
                    "expected": file_path,
                    "found": found_paths,
                    "fix": f"Update README.md to reference correct path: {found_paths[0]}",
                }
            else:
                issue = {
                    "type": "missing_file",
                    "expected": file_path,
                    "found": [],
                    "fix": f"Create missing file: {file_path}",
                }

            issues.append(issue)
            print(f"❌ {file_path} - {'MOVED' if found_paths else 'MISSING'}")
            if found_paths:
                print(f"   Found at: {', '.join(found_paths)}")
        else:
            print(f"✅ {file_path}")

    # Check key documentation cross-references
    print("\n🔍 Checking key documentation cross-references...")

    key_docs = [
        "docs/index.md",
        "README.md",
        "docs/getting-started/README.md",
        "docs/developer-guides/README.md",
    ]

    for doc_path in key_docs:
        full_path = project_root / doc_path
        if full_path.exists():
            print(f"\n📄 Checking {doc_path}...")
            issues.extend(check_doc_links(full_path, project_root))

    return issues


def check_doc_links(file_path, project_root):
    """Check links in a specific document"""
    issues = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except:
        return issues

    # Find markdown links
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for match in link_pattern.finditer(content):
        link_text = match.group(1)
        link_target = match.group(2)

        # Skip external URLs
        if link_target.startswith(("http://", "https://", "mailto:")):
            continue

        # Skip anchors
        if link_target.startswith("#"):
            continue

        # Remove fragment/anchor from target
        clean_target = link_target.split("#")[0]
        if not clean_target:
            continue

        # Check if target exists
        target_found = False

        # Try relative to current file
        relative_path = file_path.parent / clean_target
        if relative_path.exists():
            target_found = True

        # Try relative to project root
        if not target_found:
            root_path = project_root / clean_target
            if root_path.exists():
                target_found = True

        # Try docs directory
        if not target_found:
            docs_path = project_root / "docs" / clean_target
            if docs_path.exists():
                target_found = True

        if not target_found:
            line_num = content[: match.start()].count("\n") + 1
            issue = {
                "type": "broken_link",
                "source": str(file_path.relative_to(project_root)),
                "line": line_num,
                "text": link_text,
                "target": link_target,
                "fix": f"Check if target exists or update link",
            }
            issues.append(issue)
            print(f"   ❌ Line {line_num}: [{link_text}]({link_target})")

    return issues


def main():
    print("🔍 Quick Documentation Link Check")
    print("=" * 50)

    issues = check_critical_links()

    print(f"\n📊 Summary:")
    print(f"   Total issues found: {len(issues)}")

    if issues:
        print(f"\n🔧 Recommended fixes:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue['fix']}")
    else:
        print("✅ No critical issues found!")

    return len(issues)


if __name__ == "__main__":
    exit(main())
