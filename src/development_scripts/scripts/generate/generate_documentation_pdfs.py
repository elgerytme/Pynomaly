#!/usr/bin/env python3
"""
Generate PDF versions of anomaly_detection documentation.

This script converts all Markdown documentation files to PDF format
with professional styling and formatting.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import markdown
    from markdown.extensions import codehilite, fenced_code, tables, toc
    from weasyprint import CSS, HTML
except ImportError:
    print("Error: Required packages missing. Install with:")
    print("pip install markdown weasyprint")
    sys.exit(1)


class DocumentationPDFGenerator:
    """Generate PDF versions of anomaly_detection documentation."""

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.output_dir = self.docs_dir / "pdfs"
        self.output_dir.mkdir(exist_ok=True)

        # Setup Markdown processor
        self.md = markdown.Markdown(
            extensions=[
                "toc",
                "tables",
                "codehilite",
                "fenced_code",
                "attr_list",
                "def_list",
                "footnotes",
            ],
            extension_configs={
                "toc": {"permalink": True},
                "codehilite": {"css_class": "highlight"},
            },
        )

        # Define CSS styles
        self.css_styles = self.create_pdf_styles()

    def create_pdf_styles(self) -> str:
        """Create CSS styles for PDF generation."""
        return """
        @page {
            size: A4;
            margin: 1in;
            @top-center {
                content: "anomaly_detection Documentation";
                font-family: Arial, sans-serif;
                font-size: 10pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-family: Arial, sans-serif;
                font-size: 10pt;
                color: #666;
            }
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            font-size: 11pt;
        }

        h1 {
            color: #0066cc;
            font-size: 24pt;
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 20pt;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10pt;
        }

        h2 {
            color: #0066cc;
            font-size: 18pt;
            font-weight: bold;
            margin-top: 24pt;
            margin-bottom: 12pt;
            border-bottom: 1px solid #ccc;
            padding-bottom: 6pt;
        }

        h3 {
            color: #333;
            font-size: 14pt;
            font-weight: bold;
            margin-top: 18pt;
            margin-bottom: 10pt;
        }

        h4 {
            color: #333;
            font-size: 12pt;
            font-weight: bold;
            margin-top: 14pt;
            margin-bottom: 8pt;
        }

        p {
            margin-bottom: 10pt;
            text-align: justify;
        }

        ul, ol {
            margin-bottom: 12pt;
            padding-left: 20pt;
        }

        li {
            margin-bottom: 4pt;
        }

        code {
            background-color: #f5f5f5;
            padding: 2pt 4pt;
            border-radius: 2pt;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }

        pre {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4pt;
            padding: 12pt;
            margin: 12pt 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
        }

        pre code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12pt 0;
            font-size: 10pt;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 6pt 8pt;
            text-align: left;
        }

        th {
            background-color: #f5f5f5;
            font-weight: bold;
            color: #333;
        }

        tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        blockquote {
            margin: 12pt 0;
            padding: 12pt 16pt;
            background-color: #f0f7ff;
            border-left: 4px solid #0066cc;
            font-style: italic;
        }

        .toc {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 4pt;
            padding: 16pt;
            margin: 20pt 0;
        }

        .toc h2 {
            margin-top: 0;
            color: #333;
            border-bottom: none;
        }

        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }

        .toc li {
            margin-bottom: 6pt;
        }

        .toc a {
            text-decoration: none;
            color: #0066cc;
        }

        .toc a:hover {
            text-decoration: underline;
        }

        .highlight {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4pt;
            padding: 12pt;
            margin: 12pt 0;
        }

        .page-break {
            page-break-before: always;
        }

        .no-break {
            page-break-inside: avoid;
        }

        .document-title {
            text-align: center;
            margin-bottom: 30pt;
        }

        .document-info {
            background-color: #f0f7ff;
            border: 1px solid #0066cc;
            border-radius: 4pt;
            padding: 16pt;
            margin: 20pt 0;
        }

        .document-info h3 {
            margin-top: 0;
            color: #0066cc;
        }
        """

    def add_document_header(self, title: str, content: str) -> str:
        """Add document header with title and metadata."""
        header = f"""
        <div class="document-title">
            <h1>{title}</h1>
        </div>

        <div class="document-info">
            <h3>Document Information</h3>
            <p><strong>Title:</strong> {title}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Version:</strong> 1.0</p>
            <p><strong>Project:</strong> anomaly_detection - State-of-the-Art Anomaly Detection Platform</p>
        </div>

        <div class="page-break"></div>
        """

        return header + content

    def convert_markdown_to_html(self, markdown_content: str, title: str) -> str:
        """Convert Markdown content to HTML with proper styling."""
        # Convert markdown to HTML
        html_content = self.md.convert(markdown_content)

        # Add document header
        html_content = self.add_document_header(title, html_content)

        # Create complete HTML document
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                {self.css_styles}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        return html_doc

    def generate_pdf_from_markdown(
        self, markdown_file: Path, output_file: Path
    ) -> bool:
        """Generate PDF from a Markdown file."""
        try:
            print(f"Converting {markdown_file.name} to PDF...")

            # Read Markdown content
            with open(markdown_file, encoding="utf-8") as f:
                markdown_content = f.read()

            # Get title from filename
            title = markdown_file.stem.replace("-", " ").replace("_", " ").title()
            if title.startswith("0"):
                # Remove numbering prefix
                title = title[2:]

            # Convert to HTML
            html_content = self.convert_markdown_to_html(markdown_content, title)

            # Generate PDF
            html_doc = HTML(string=html_content)
            css = CSS(string=self.css_styles)
            html_doc.write_pdf(str(output_file), stylesheets=[css])

            print(f"✅ Successfully generated: {output_file}")
            return True

        except Exception as e:
            print(f"❌ Error generating PDF for {markdown_file.name}: {e}")
            return False

    def generate_combined_pdf(
        self, markdown_files: list[Path], output_file: Path
    ) -> bool:
        """Generate a single combined PDF from multiple Markdown files."""
        try:
            print("Generating combined documentation PDF...")

            combined_content = ""

            # Add cover page
            cover_content = f"""
            # anomaly_detection Documentation
            ## Complete Guide to State-of-the-Art Anomaly Detection

            <div class="document-info">
                <h3>Documentation Package</h3>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Version:</strong> 1.0</p>
                <p><strong>Documents Included:</strong> {len(markdown_files)}</p>
            </div>

            ### Table of Contents

            """

            # Add table of contents
            for i, md_file in enumerate(markdown_files, 1):
                title = md_file.stem.replace("-", " ").replace("_", " ").title()
                if title.startswith("0"):
                    title = title[2:]
                cover_content += f"{i}. {title}\n"

            cover_content += "\n<div class='page-break'></div>\n\n"
            combined_content += cover_content

            # Process each Markdown file
            for md_file in markdown_files:
                print(f"Processing {md_file.name}...")

                with open(md_file, encoding="utf-8") as f:
                    content = f.read()

                # Add page break before each new document
                combined_content += f"<div class='page-break'></div>\n\n{content}\n\n"

            # Convert combined content to HTML
            title = "anomaly_detection Complete Documentation"
            html_content = self.convert_markdown_to_html(combined_content, title)

            # Generate PDF
            html_doc = HTML(string=html_content)
            css = CSS(string=self.css_styles)
            html_doc.write_pdf(str(output_file), stylesheets=[css])

            print(f"✅ Successfully generated combined PDF: {output_file}")
            return True

        except Exception as e:
            print(f"❌ Error generating combined PDF: {e}")
            return False

    def generate_all_pdfs(self) -> dict[str, Any]:
        """Generate PDFs for all Markdown files."""
        results = {
            "individual_pdfs": [],
            "combined_pdf": None,
            "total_files": 0,
            "successful": 0,
            "failed": 0,
        }

        # Find all Markdown files in the docs directory
        markdown_files = list(self.docs_dir.glob("*.md"))
        markdown_files.sort()  # Sort for consistent ordering

        results["total_files"] = len(markdown_files)

        if not markdown_files:
            print("No Markdown files found in the documentation directory.")
            return results

        # Generate individual PDFs
        for md_file in markdown_files:
            output_file = self.output_dir / f"{md_file.stem}.pdf"

            if self.generate_pdf_from_markdown(md_file, output_file):
                results["individual_pdfs"].append(str(output_file))
                results["successful"] += 1
            else:
                results["failed"] += 1

        # Generate combined PDF
        combined_output = self.output_dir / "Monorepo_Complete_Documentation.pdf"
        if self.generate_combined_pdf(markdown_files, combined_output):
            results["combined_pdf"] = str(combined_output)

        return results


def main():
    """Main function to generate PDFs."""
    parser = argparse.ArgumentParser(
        description="Generate PDF versions of anomaly_detection documentation"
    )
    parser.add_argument(
        "--docs-dir",
        default="/mnt/c/Users/andre/anomaly_detection/docs/comprehensive",
        help="Directory containing Markdown documentation files",
    )
    parser.add_argument(
        "--output-dir", help="Output directory for PDF files (default: {docs-dir}/pdfs)"
    )

    args = parser.parse_args()

    # Set up directories
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"Error: Documentation directory not found: {docs_dir}")
        sys.exit(1)

    # Generate PDFs
    generator = DocumentationPDFGenerator(str(docs_dir))

    if args.output_dir:
        generator.output_dir = Path(args.output_dir)
        generator.output_dir.mkdir(parents=True, exist_ok=True)

    results = generator.generate_all_pdfs()

    # Print summary
    print("\n" + "=" * 60)
    print("PDF GENERATION SUMMARY")
    print("=" * 60)
    print(f"📁 Documentation directory: {docs_dir}")
    print(f"📁 Output directory: {generator.output_dir}")
    print(f"📄 Total Markdown files: {results['total_files']}")
    print(f"✅ Successfully converted: {results['successful']}")
    print(f"❌ Failed conversions: {results['failed']}")

    if results["individual_pdfs"]:
        print("\n📋 Individual PDFs generated:")
        for pdf_file in results["individual_pdfs"]:
            print(f"   • {Path(pdf_file).name}")

    if results["combined_pdf"]:
        print(f"\n📚 Combined PDF: {Path(results['combined_pdf']).name}")

    print(f"\n🎯 All PDF files are available in: {generator.output_dir}")

    return results


if __name__ == "__main__":
    main()
