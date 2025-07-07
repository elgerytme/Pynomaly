# PDF Generation Instructions

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Project

---


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
