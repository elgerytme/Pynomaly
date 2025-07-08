#!/usr/bin/env python3
"""
ADR Validation Script

This script validates Architectural Decision Records (ADRs) to ensure they:
1. Follow the correct filename pattern (ADR-###-slug.md)
2. Have all required template sections
3. Have valid status values
4. Maintain sequential numbering
"""

import os
import re
import sys
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class ADRValidationError(Exception):
    """Custom exception for ADR validation errors."""
    pass


class ADRValidator:
    """Validates ADR files according to project standards."""
    
    VALID_STATUSES = {'PROPOSED', 'ACCEPTED', 'REJECTED', 'SUPERSEDED', 'DEPRECATED'}
    
    REQUIRED_SECTIONS = [
        'Context',
        'Decision', 
        'Rationale',
        'Alternatives Considered',
        'Consequences',
        'Implementation',
        'Monitoring',
        'Related Decisions',
        'References'
    ]
    
    def __init__(self, adr_directory: str = "docs/developer-guides/architecture/adr"):
        """Initialize the validator with the ADR directory path."""
        self.adr_directory = Path(adr_directory)
        self.errors: List[str] = []
        
    def validate_filename(self, filepath: Path) -> Optional[int]:
        """
        Validate ADR filename follows the pattern ADR-###-slug.md
        
        Returns:
            The ADR number if valid, None otherwise
        """
        filename = filepath.name
        
        # Check if it's an ADR file
        if not filename.startswith('ADR-'):
            return None
            
        # Pattern: ADR-###-slug.md
        pattern = r'^ADR-(\d{3})-([a-z0-9-]+)\.md$'
        match = re.match(pattern, filename)
        
        if not match:
            self.errors.append(f"❌ Invalid filename format: {filename}")
            self.errors.append(f"   Expected pattern: ADR-###-slug.md")
            self.errors.append(f"   Example: ADR-001-database-choice.md")
            return None
            
        adr_number = int(match.group(1))
        slug = match.group(2)
        
        # Additional validations
        if adr_number == 0:
            self.errors.append(f"❌ ADR number cannot be 000: {filename}")
            return None
            
        # Validate slug (kebab-case)
        if not re.match(r'^[a-z0-9-]+$', slug):
            self.errors.append(f"❌ Invalid slug format in {filename}")
            self.errors.append(f"   Slug must be lowercase kebab-case: {slug}")
            return None
            
        return adr_number
    
    def validate_content(self, filepath: Path) -> Dict[str, str]:
        """
        Validate ADR content has all required sections.
        
        Returns:
            Dictionary with extracted metadata
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"❌ Error reading file {filepath}: {e}")
            return {}
            
        metadata = {}
        
        # Extract status
        status_match = re.search(r'\*\*Status:\*\*\s+([A-Z]+)', content)
        if status_match:
            status = status_match.group(1)
            if status not in self.VALID_STATUSES:
                self.errors.append(f"❌ Invalid status '{status}' in {filepath}")
                self.errors.append(f"   Valid statuses: {', '.join(sorted(self.VALID_STATUSES))}")
            else:
                metadata['status'] = status
        else:
            self.errors.append(f"❌ Missing or invalid status format in {filepath}")
            self.errors.append(f"   Expected: **Status:** PROPOSED|ACCEPTED|REJECTED|SUPERSEDED|DEPRECATED")
        
        # Extract date
        date_match = re.search(r'\*\*Date:\*\*\s+(\d{4}-\d{2}-\d{2})', content)
        if date_match:
            metadata['date'] = date_match.group(1)
        else:
            self.errors.append(f"❌ Missing or invalid date format in {filepath}")
            self.errors.append(f"   Expected: **Date:** YYYY-MM-DD")
            
        # Extract author
        author_match = re.search(r'\*\*Author:\*\*\s+(@[\w-]+)', content)
        if author_match:
            metadata['author'] = author_match.group(1)
        else:
            self.errors.append(f"❌ Missing or invalid author format in {filepath}")
            self.errors.append(f"   Expected: **Author:** @username")
        
        # Check for required sections
        for section in self.REQUIRED_SECTIONS:
            section_pattern = rf'^## {re.escape(section)}$'
            if not re.search(section_pattern, content, re.MULTILINE):
                self.errors.append(f"❌ Missing required section '{section}' in {filepath}")
                continue
                
            # Check if section is empty (only contains comments)
            section_content = self.extract_section_content(content, section)
            if self.is_section_empty(section_content):
                self.errors.append(f"❌ Section '{section}' is empty in {filepath}")
                self.errors.append(f"   Please fill in the content or remove placeholder comments")
        
        return metadata
    
    def extract_section_content(self, content: str, section: str) -> str:
        """Extract content of a specific section."""
        pattern = rf'^## {re.escape(section)}$\n(.*?)(?=^## |\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        return match.group(1) if match else ""
    
    def is_section_empty(self, section_content: str) -> bool:
        """Check if a section is empty (only contains comments and whitespace)."""
        # Remove HTML comments
        content_without_comments = re.sub(r'<!--.*?-->', '', section_content, flags=re.DOTALL)
        
        # Remove whitespace and newlines
        content_without_whitespace = re.sub(r'\s+', '', content_without_comments)
        
        return len(content_without_whitespace) == 0
    
    def validate_sequential_numbering(self, adr_numbers: List[int]) -> None:
        """Validate that ADR numbers are sequential with no gaps."""
        if not adr_numbers:
            return
            
        sorted_numbers = sorted(adr_numbers)
        
        # Check for gaps
        for i in range(len(sorted_numbers) - 1):
            current = sorted_numbers[i]
            next_num = sorted_numbers[i + 1]
            
            if next_num - current != 1:
                self.errors.append(f"❌ Gap in ADR numbering: {current} → {next_num}")
                self.errors.append(f"   ADR numbers must be sequential")
                
        # Check starting number
        if sorted_numbers[0] != 1:
            self.errors.append(f"❌ ADR numbering should start at 001, found: {sorted_numbers[0]:03d}")
    
    def validate_all(self) -> bool:
        """
        Validate all ADR files in the directory.
        
        Returns:
            True if all validations pass, False otherwise
        """
        self.errors = []
        
        # Find all ADR files
        adr_pattern = str(self.adr_directory / "ADR-*.md")
        adr_files = glob.glob(adr_pattern)
        
        if not adr_files:
            print("ℹ️  No ADR files found to validate")
            return True
            
        print(f"🔍 Validating {len(adr_files)} ADR files...")
        
        adr_numbers = []
        file_metadata = {}
        
        for adr_file in adr_files:
            filepath = Path(adr_file)
            print(f"  Validating: {filepath.name}")
            
            # Validate filename
            adr_number = self.validate_filename(filepath)
            if adr_number:
                adr_numbers.append(adr_number)
                
            # Validate content
            metadata = self.validate_content(filepath)
            if metadata:
                file_metadata[filepath.name] = metadata
        
        # Validate sequential numbering
        self.validate_sequential_numbering(adr_numbers)
        
        # Report results
        if self.errors:
            print("\\n❌ ADR Validation Failed:")
            for error in self.errors:
                print(f"  {error}")
            return False
        else:
            print("\\n✅ All ADR validations passed!")
            return True


def main():
    """Main function to run ADR validation."""
    print("🚀 Starting ADR Validation...")
    
    validator = ADRValidator()
    
    # Check if ADR directory exists
    if not validator.adr_directory.exists():
        print(f"❌ ADR directory not found: {validator.adr_directory}")
        sys.exit(1)
    
    # Run validation
    success = validator.validate_all()
    
    if success:
        print("\\n🎉 ADR validation completed successfully!")
        sys.exit(0)
    else:
        print("\\n💥 ADR validation failed!")
        print("\\n📖 Resources:")
        print("   - ADR Template: docs/developer-guides/architecture/adr/template.md")
        print("   - ADR Guidelines: docs/developer-guides/architecture/adr/README.md")
        print("   - Contributing: docs/developer-guides/contributing/CONTRIBUTING.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
