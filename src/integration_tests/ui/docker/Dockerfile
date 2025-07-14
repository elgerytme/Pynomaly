# Multi-stage Dockerfile for UI testing with Chrome and Firefox
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for UI testing
RUN pip install --no-cache-dir \
    playwright \
    pytest \
    pytest-playwright \
    pytest-xdist \
    pytest-html \
    pytest-json-report

# Install browsers
RUN playwright install chrome firefox

# Set environment variables
ENV PYTHONPATH=/app
ENV HEADLESS=true
ENV BROWSER=chrome
ENV TAKE_SCREENSHOTS=true
ENV RECORD_VIDEOS=true
ENV RECORD_TRACES=false

# Create directories for test artifacts
RUN mkdir -p /app/test_reports/screenshots /app/test_reports/videos /app/test_reports/traces

# Copy test files
COPY tests/ /app/tests/

# Set the default command
CMD ["python", "-m", "pytest", "tests/ui/", "-v", "--html=test_reports/report.html", "--json-report", "--json-report-file=test_reports/report.json"]
