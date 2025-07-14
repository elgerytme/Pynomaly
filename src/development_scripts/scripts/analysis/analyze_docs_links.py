#!/usr/bin/env python3
"""
Comprehensive Documentation Cross-Linking Analysis Tool

This script analyzes the documentation structure in the /docs/ directory to:
1. Identify existing cross-links and patterns
2. Catalog broken or missing links
3. Map documentation structure and relationships
4. Identify high-value linking opportunities
5. Document current linking conventions
6. Suggest improvements and standards
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


class DocumentationAnalyzer:
    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path)
        self.all_docs = {}  # path -> content
        self.links = {}  # source_path -> [(link_text, target_path, line_number)]
        self.broken_links = []  # [(source_path, link_text, target_path, reason)]
        self.link_patterns = Counter()  # Track common linking patterns
        self.directory_structure = {}
        self.cross_references = defaultdict(list)  # target_path -> [source_paths]

    def scan_documentation(self):
        """Scan all markdown files in the docs directory"""
        print("🔍 Scanning documentation files...")

        for md_file in self.docs_path.rglob("*.md"):
            rel_path = str(md_file.relative_to(self.docs_path))
            try:
                with open(md_file, encoding="utf-8") as f:
                    content = f.read()
                    self.all_docs[rel_path] = content
                    print(f"  ✓ {rel_path}")
            except Exception as e:
                print(f"  ⚠️  Error reading {rel_path}: {e}")

        print(f"📄 Found {len(self.all_docs)} documentation files")

    def extract_links(self):
        """Extract all markdown links from documentation files"""
        print("\n🔗 Extracting cross-links...")

        # Regex patterns for different link types
        link_patterns = [
            # Standard markdown links [text](path)
            r"\[([^\]]+)\]\(([^)]+\.md[^)]*)\)",
            # Reference-style links [text][ref]
            r"\[([^\]]+)\]\[([^\]]+)\]",
            # Direct links to .md files
            r"(?:^|\s)([^\s]+\.md)(?:\s|$)",
        ]

        for doc_path, content in self.all_docs.items():
            self.links[doc_path] = []
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                # Standard markdown links
                for match in re.finditer(r"\[([^\]]+)\]\(([^)]+\.md[^)]*)\)", line):
                    link_text = match.group(1)
                    target_path = match.group(2)
                    self.links[doc_path].append((link_text, target_path, line_num))

                    # Track pattern
                    if target_path.startswith("../"):
                        self.link_patterns["relative_parent"] += 1
                    elif target_path.startswith("./"):
                        self.link_patterns["relative_current"] += 1
                    elif "/" in target_path:
                        self.link_patterns["directory_path"] += 1
                    else:
                        self.link_patterns["simple_filename"] += 1

        total_links = sum(len(links) for links in self.links.values())
        print(f"🔗 Found {total_links} cross-links across {len(self.links)} files")

    def validate_links(self):
        """Validate all links and identify broken ones"""
        print("\n🔍 Validating links...")

        for source_path, links in self.links.items():
            source_dir = Path(source_path).parent

            for link_text, target_path, line_num in links:
                # Skip external links and anchors
                if target_path.startswith("http"):
                    continue
                if "#" in target_path:
                    target_path = target_path.split("#")[0]
                if not target_path:
                    continue

                # Resolve relative paths
                if target_path.startswith("../") or target_path.startswith("./"):
                    try:
                        resolved_path = (
                            self.docs_path / source_dir / target_path
                        ).resolve()
                        relative_to_docs = resolved_path.relative_to(
                            self.docs_path.resolve()
                        )
                        target_doc_path = str(relative_to_docs)
                    except (ValueError, OSError):
                        # If path resolution fails, try simple string manipulation
                        target_doc_path = target_path
                else:
                    # Assume it's relative to current directory
                    target_doc_path = (
                        str(source_dir / target_path)
                        if source_dir != Path(".")
                        else target_path
                    )

                # Check if target exists
                if target_doc_path not in self.all_docs:
                    # Try to find the file by name
                    possible_matches = [
                        path
                        for path in self.all_docs.keys()
                        if path.endswith(target_path)
                    ]
                    if possible_matches:
                        reason = f"Path mismatch - found candidates: {', '.join(possible_matches)}"
                    else:
                        reason = "File not found"

                    self.broken_links.append(
                        (source_path, link_text, target_path, reason)
                    )
                else:
                    # Valid link - record cross-reference
                    self.cross_references[target_doc_path].append(source_path)

        print(f"❌ Found {len(self.broken_links)} broken links")

    def analyze_structure(self):
        """Analyze documentation structure and relationships"""
        print("\n📊 Analyzing documentation structure...")

        # Build directory hierarchy
        self.directory_structure = {}
        for doc_path in self.all_docs.keys():
            parts = doc_path.split("/")
            current = self.directory_structure
            for part in parts[:-1]:  # Exclude filename
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Add file to structure
            filename = parts[-1]
            current[filename] = {
                "size": len(self.all_docs[doc_path]),
                "links_out": len(self.links.get(doc_path, [])),
                "links_in": len(self.cross_references.get(doc_path, [])),
            }

    def identify_opportunities(self):
        """Identify high-value cross-linking opportunities"""
        print("\n💡 Identifying cross-linking opportunities...")

        opportunities = []

        # 1. Files with many incoming links but few outgoing links
        for doc_path, content in self.all_docs.items():
            incoming = len(self.cross_references.get(doc_path, []))
            outgoing = len(self.links.get(doc_path, []))

            if incoming > 3 and outgoing < 2:
                opportunities.append(
                    {
                        "type": "hub_document",
                        "file": doc_path,
                        "description": f"High-traffic document ({incoming} incoming links) with few outgoing links ({outgoing})",
                        "priority": "high",
                    }
                )

        # 2. Related files that don't link to each other
        getting_started = [
            path for path in self.all_docs.keys() if "getting-started" in path
        ]
        user_guides = [path for path in self.all_docs.keys() if "user-guides" in path]
        developer_guides = [
            path for path in self.all_docs.keys() if "developer-guides" in path
        ]
        examples = [path for path in self.all_docs.keys() if "examples" in path]

        # Check for missing connections between sections
        for gs_doc in getting_started:
            gs_links = [link[1] for link in self.links.get(gs_doc, [])]
            missing_user_guides = [
                ug for ug in user_guides if not any(ug in link for link in gs_links)
            ]
            if missing_user_guides:
                opportunities.append(
                    {
                        "type": "section_connection",
                        "file": gs_doc,
                        "description": f"Getting started doc could link to user guides: {missing_user_guides[:3]}",
                        "priority": "medium",
                    }
                )

        # 3. Orphaned documents (no incoming links)
        for doc_path in self.all_docs.keys():
            if doc_path not in self.cross_references and doc_path != "index.md":
                opportunities.append(
                    {
                        "type": "orphaned_document",
                        "file": doc_path,
                        "description": "Document has no incoming links",
                        "priority": "medium",
                    }
                )

        return opportunities

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📈 Generating comprehensive report...")

        opportunities = self.identify_opportunities()

        report = {
            "summary": {
                "total_documents": len(self.all_docs),
                "total_links": sum(len(links) for links in self.links.values()),
                "broken_links": len(self.broken_links),
                "cross_referenced_docs": len(self.cross_references),
                "orphaned_docs": len(
                    [opp for opp in opportunities if opp["type"] == "orphaned_document"]
                ),
            },
            "link_patterns": dict(self.link_patterns),
            "broken_links": self.broken_links,
            "cross_references": {k: v for k, v in self.cross_references.items()},
            "opportunities": opportunities,
            "directory_structure": self.directory_structure,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self):
        """Generate specific recommendations for improvement"""
        recommendations = []

        # Link pattern recommendations
        if (
            self.link_patterns["relative_parent"]
            > self.link_patterns["relative_current"]
        ):
            recommendations.append(
                {
                    "category": "conventions",
                    "priority": "medium",
                    "description": "Consider standardizing on relative current directory links (./file.md) over parent directory links (../file.md) where possible",
                }
            )

        # Navigation recommendations
        if len(self.broken_links) > 0:
            recommendations.append(
                {
                    "category": "maintenance",
                    "priority": "high",
                    "description": f"Fix {len(self.broken_links)} broken links to improve user experience",
                }
            )

        # Structure recommendations
        main_sections = [
            "getting-started",
            "user-guides",
            "developer-guides",
            "examples",
            "reference",
        ]
        for section in main_sections:
            section_docs = [path for path in self.all_docs.keys() if section in path]
            section_links = sum(len(self.links.get(doc, [])) for doc in section_docs)
            if section_docs and section_links < len(section_docs):
                recommendations.append(
                    {
                        "category": "navigation",
                        "priority": "medium",
                        "description": f"Improve internal linking within {section} section (avg {section_links/len(section_docs):.1f} links per doc)",
                    }
                )

        return recommendations


def main():
    docs_path = "/mnt/c/Users/andre/Pynomaly/docs"

    print("🔍 Pynomaly Documentation Cross-Linking Analysis")
    print("=" * 50)

    analyzer = DocumentationAnalyzer(docs_path)

    # Run analysis
    analyzer.scan_documentation()
    analyzer.extract_links()
    analyzer.validate_links()
    analyzer.analyze_structure()

    # Generate report
    report = analyzer.generate_report()

    # Save detailed report
    with open("/mnt/c/Users/andre/Pynomaly/docs_cross_linking_analysis.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"📄 Total Documents: {report['summary']['total_documents']}")
    print(f"🔗 Total Cross-Links: {report['summary']['total_links']}")
    print(f"❌ Broken Links: {report['summary']['broken_links']}")
    print(f"🔄 Cross-Referenced Docs: {report['summary']['cross_referenced_docs']}")
    print(f"🏝️  Orphaned Documents: {report['summary']['orphaned_docs']}")

    print("\n🔗 LINK PATTERNS:")
    for pattern, count in report["link_patterns"].items():
        print(f"  {pattern}: {count}")

    if report["broken_links"]:
        print("\n❌ BROKEN LINKS (showing first 10):")
        for i, (source, text, target, reason) in enumerate(report["broken_links"][:10]):
            print(f"  {i+1}. {source}: '{text}' -> '{target}' ({reason})")

    print("\n💡 IMPROVEMENT OPPORTUNITIES (showing first 10):")
    for i, opp in enumerate(report["opportunities"][:10]):
        print(f"  {i+1}. [{opp['priority'].upper()}] {opp['type']}: {opp['file']}")
        print(f"      {opp['description']}")

    print("\n📋 RECOMMENDATIONS:")
    for i, rec in enumerate(report["recommendations"]):
        print(
            f"  {i+1}. [{rec['priority'].upper()}] {rec['category']}: {rec['description']}"
        )

    print("\n📄 Full analysis saved to: docs_cross_linking_analysis.json")
    print("\n🎯 Next Steps:")
    print("  1. Fix broken links identified in the analysis")
    print("  2. Add cross-references for orphaned documents")
    print("  3. Enhance hub documents with outgoing links")
    print("  4. Standardize linking conventions")
    print("  5. Implement suggested navigation improvements")


if __name__ == "__main__":
    main()
