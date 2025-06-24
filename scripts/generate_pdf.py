#!/usr/bin/env python3
"""
Generate PDF version of Banking Anomaly Detection Guide
"""

import sys
import os
from pathlib import Path

# Try different PDF generation approaches
def generate_pdf_from_markdown():
    """Generate PDF from the markdown file using available tools."""
    
    # Path to the markdown file
    markdown_file = Path(__file__).parent.parent / "docs" / "Banking_Anomaly_Detection_Guide.md"
    pdf_file = Path(__file__).parent.parent / "docs" / "Banking_Anomaly_Detection_Guide.pdf"
    
    if not markdown_file.exists():
        print(f"Markdown file not found: {markdown_file}")
        return False
    
    # Try different approaches in order of preference
    approaches = [
        ("pandoc", f"pandoc '{markdown_file}' -o '{pdf_file}' --pdf-engine=pdflatex"),
        ("pandoc-xelatex", f"pandoc '{markdown_file}' -o '{pdf_file}' --pdf-engine=xelatex"),
        ("pandoc-wkhtmltopdf", f"pandoc '{markdown_file}' -o '{pdf_file}' --pdf-engine=wkhtmltopdf"),
        ("markdown-to-html-to-pdf", None),  # Custom approach
    ]
    
    for approach_name, command in approaches:
        if command:
            try:
                result = os.system(command)
                if result == 0 and pdf_file.exists():
                    print(f"Successfully generated PDF using {approach_name}: {pdf_file}")
                    return True
            except Exception as e:
                print(f"Failed with {approach_name}: {e}")
                continue
    
    # If all else fails, create a simple text-based PDF instruction file
    create_pdf_instructions()
    return False


def create_pdf_instructions():
    """Create instructions for manual PDF generation."""
    
    instructions_file = Path(__file__).parent.parent / "docs" / "PDF_Generation_Instructions.md"
    
    instructions = """# PDF Generation Instructions

Since automated PDF generation tools are not available in this environment, please follow these steps to create a PDF version of the Banking Anomaly Detection Guide:

## Option 1: Using Online Markdown to PDF Converter
1. Open the file `Banking_Anomaly_Detection_Guide.md` in a text editor
2. Copy the entire content
3. Visit an online Markdown to PDF converter such as:
   - https://md2pdf.netlify.app/
   - https://www.markdowntopdf.com/
   - https://dillinger.io/ (export as PDF)
4. Paste the content and download the PDF

## Option 2: Using Pandoc (if available)
```bash
# Install pandoc if not available
sudo apt-get install pandoc texlive-latex-recommended

# Generate PDF
pandoc Banking_Anomaly_Detection_Guide.md -o Banking_Anomaly_Detection_Guide.pdf --pdf-engine=pdflatex
```

## Option 3: Using Browser Print-to-PDF
1. Open the HTML presentation file `Banking_Anomaly_Detection_Slides.html` in a web browser
2. Use Ctrl+P (or Cmd+P on Mac) to open print dialog
3. Select "Save as PDF" as the destination
4. Adjust print settings as needed and save

## Option 4: Using Microsoft Word/Google Docs
1. Copy the content from `Banking_Anomaly_Detection_Guide.md`
2. Paste into Microsoft Word or Google Docs
3. Apply appropriate formatting (headings, bullet points, etc.)
4. Export/Save as PDF

## Option 5: Using VS Code with Markdown PDF Extension
1. Install the "Markdown PDF" extension in VS Code
2. Open `Banking_Anomaly_Detection_Guide.md`
3. Right-click and select "Markdown PDF: Export (pdf)"

The resulting PDF will be suitable for business presentations and distribution to banking stakeholders.
"""
    
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"Created PDF generation instructions: {instructions_file}")


def main():
    """Main function to generate PDF."""
    print("Attempting to generate PDF from Banking Anomaly Detection Guide...")
    
    success = generate_pdf_from_markdown()
    
    if not success:
        print("Automated PDF generation not available in this environment.")
        print("Created instructions for manual PDF generation.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())