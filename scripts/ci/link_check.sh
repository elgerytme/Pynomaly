#!/bin/bash

# Link Checker for CI/CD Pipeline
# Validates documentation links and references after cleanup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Running Documentation Link Check${NC}"
echo "=========================================="

# Check if Python link checker exists
if [ ! -f "scripts/analysis/quick_link_check.py" ]; then
    echo -e "${RED}❌ Link checker script not found${NC}"
    exit 1
fi

# Run Python link checker
echo -e "${GREEN}Running Python link checker...${NC}"
python scripts/analysis/quick_link_check.py

LINK_CHECK_EXIT_CODE=$?

# Check MkDocs configuration
echo -e "\n${GREEN}🔧 Validating MkDocs configuration...${NC}"
if [ -f "config/docs/mkdocs.yml" ]; then
    echo -e "${GREEN}✅ MkDocs configuration found${NC}"
    
    # Check if mkdocs is installed
    if command -v mkdocs &> /dev/null; then
        echo -e "${GREEN}Validating MkDocs configuration...${NC}"
        mkdocs build --config-file config/docs/mkdocs.yml --site-dir temp_site --quiet
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ MkDocs configuration is valid${NC}"
            rm -rf temp_site
        else
            echo -e "${RED}❌ MkDocs configuration has errors${NC}"
            rm -rf temp_site
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️ MkDocs not installed, skipping validation${NC}"
    fi
else
    echo -e "${RED}❌ MkDocs configuration not found${NC}"
    exit 1
fi

# Check for common documentation issues
echo -e "\n${GREEN}🔍 Checking for common documentation issues...${NC}"

# Check for orphaned files
echo -e "${GREEN}Checking for orphaned documentation files...${NC}"
orphaned_files=$(find docs -name "*.md" -not -path "*/archive/*" -not -path "*/.*" | while read file; do
    # Check if file is referenced in any other documentation
    file_name=$(basename "$file")
    file_path_from_docs=${file#docs/}
    
    # Search for references to this file
    references=$(grep -r "$file_name\|$file_path_from_docs" docs/ --include="*.md" --exclude-dir=archive | grep -v "^$file:" | wc -l)
    
    if [ "$references" -eq 0 ]; then
        echo "$file"
    fi
done)

if [ -n "$orphaned_files" ]; then
    echo -e "${YELLOW}⚠️ Found potentially orphaned files:${NC}"
    echo "$orphaned_files"
else
    echo -e "${GREEN}✅ No orphaned files found${NC}"
fi

# Check for missing README files in directories
echo -e "\n${GREEN}Checking for missing README files...${NC}"
missing_readmes=$(find docs -type d -not -path "*/archive/*" -not -path "*/.*" | while read dir; do
    if [ ! -f "$dir/README.md" ] && [ ! -f "$dir/index.md" ]; then
        # Check if directory has any .md files
        if [ "$(find "$dir" -maxdepth 1 -name "*.md" | wc -l)" -gt 0 ]; then
            echo "$dir"
        fi
    fi
done)

if [ -n "$missing_readmes" ]; then
    echo -e "${YELLOW}⚠️ Directories missing README.md:${NC}"
    echo "$missing_readmes"
else
    echo -e "${GREEN}✅ All directories have README files${NC}"
fi

# Summary
echo -e "\n${GREEN}📊 Link Check Summary${NC}"
echo "======================="

if [ $LINK_CHECK_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All critical links are valid${NC}"
else
    echo -e "${RED}❌ Found $LINK_CHECK_EXIT_CODE link issues${NC}"
fi

if [ -n "$orphaned_files" ]; then
    echo -e "${YELLOW}⚠️ Found orphaned files (review recommended)${NC}"
fi

if [ -n "$missing_readmes" ]; then
    echo -e "${YELLOW}⚠️ Found directories missing README files${NC}"
fi

# Exit with error code if critical issues found
if [ $LINK_CHECK_EXIT_CODE -ne 0 ]; then
    echo -e "\n${RED}❌ Link check failed - please fix broken links${NC}"
    exit $LINK_CHECK_EXIT_CODE
fi

echo -e "\n${GREEN}🎉 Documentation link check completed successfully${NC}"
exit 0
