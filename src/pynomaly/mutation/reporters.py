"""
Reporters module for mutation testing.

This module translates mutmut junitxml output into Python objects and provides
HTML reporting capabilities.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re


@dataclass
class MutationResult:
    """Represents a single mutation test result."""
    
    id: str
    status: str
    mutant_code: str
    original_code: str
    filename: str
    line_number: int
    test_that_killed: Optional[str] = None
    
    @property
    def is_killed(self) -> bool:
        """Check if the mutation was killed (test failed)."""
        return self.status == "killed"
    
    @property
    def is_survived(self) -> bool:
        """Check if the mutation survived (test passed)."""
        return self.status == "survived"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "status": self.status,
            "mutant_code": self.mutant_code,
            "original_code": self.original_code,
            "filename": self.filename,
            "line_number": self.line_number,
            "test_that_killed": self.test_that_killed
        }


class MutationTestSuite:
    """Represents a collection of mutation test results."""
    
    def __init__(self, name: str):
        """
        Initialize the test suite.
        
        Args:
            name: Name of the test suite
        """
        self.name = name
        self.results: List[MutationResult] = []
    
    def add_result(self, result: MutationResult) -> None:
        """
        Add a mutation result to the suite.
        
        Args:
            result: MutationResult to add
        """
        self.results.append(result)
    
    @property
    def total_mutations(self) -> int:
        """Get total number of mutations."""
        return len(self.results)
    
    @property
    def killed_mutations(self) -> int:
        """Get number of killed mutations."""
        return sum(1 for result in self.results if result.is_killed)
    
    @property
    def survived_mutations(self) -> int:
        """Get number of survived mutations."""
        return sum(1 for result in self.results if result.is_survived)
    
    @property
    def mutation_score(self) -> float:
        """Calculate mutation score (percentage of killed mutations)."""
        if self.total_mutations == 0:
            return 0.0
        return round((self.killed_mutations / self.total_mutations) * 100, 2)


class JunitXmlParser:
    """Parses JUnit XML output from mutmut into Python objects."""
    
    def parse(self, xml_content: str) -> MutationTestSuite:
        """
        Parse XML content into MutationTestSuite.
        
        Args:
            xml_content: XML content as string
            
        Returns:
            MutationTestSuite instance
            
        Raises:
            ValueError: If XML format is invalid
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
        
        # Find the testsuite element
        testsuite = root.find('.//testsuite')
        if testsuite is None:
            raise ValueError("No testsuite element found in XML")
        
        suite_name = testsuite.get('name', 'unknown')
        suite = MutationTestSuite(suite_name)
        
        # Parse each testcase
        for testcase in testsuite.findall('testcase'):
            result = self._parse_testcase(testcase)
            suite.add_result(result)
        
        return suite
    
    def parse_from_file(self, xml_file: str) -> MutationTestSuite:
        """
        Parse XML from file.
        
        Args:
            xml_file: Path to XML file
            
        Returns:
            MutationTestSuite instance
        """
        with open(xml_file, 'r') as f:
            xml_content = f.read()
        
        return self.parse(xml_content)
    
    def _parse_testcase(self, testcase: ET.Element) -> MutationResult:
        """Parse a single testcase element into MutationResult."""
        mutant_id = testcase.get('name', '')
        classname = testcase.get('classname', '')
        
        # Extract filename and line number from classname
        filename, line_number = self._parse_classname(classname)
        
        # Check if mutation was killed (has failure element)
        failure = testcase.find('failure')
        if failure is not None:
            status = "killed"
            test_that_killed = failure.get('message', '')
            # Parse original and mutant code from failure text
            original_code, mutant_code = self._parse_failure_text(failure.text or '')
        else:
            status = "survived"
            test_that_killed = None
            original_code = ""
            mutant_code = ""
        
        return MutationResult(
            id=mutant_id,
            status=status,
            mutant_code=mutant_code,
            original_code=original_code,
            filename=filename,
            line_number=line_number,
            test_that_killed=test_that_killed
        )
    
    def _parse_classname(self, classname: str) -> tuple[str, int]:
        """Parse filename and line number from classname."""
        # Example: "test.py:10" -> ("test.py", 10)
        if ':' in classname:
            parts = classname.split(':')
            filename = parts[0]
            try:
                line_number = int(parts[1])
            except (ValueError, IndexError):
                line_number = 0
        else:
            filename = classname
            line_number = 0
        
        return filename, line_number
    
    def _parse_failure_text(self, failure_text: str) -> tuple[str, str]:
        """Parse original and mutant code from failure text."""
        # Example: "Original: x - 1, Mutant: x + 1"
        original_pattern = r'Original:\s*([^,]+)'
        mutant_pattern = r'Mutant:\s*(.+)'
        
        original_match = re.search(original_pattern, failure_text)
        mutant_match = re.search(mutant_pattern, failure_text)
        
        original_code = original_match.group(1).strip() if original_match else ""
        mutant_code = mutant_match.group(1).strip() if mutant_match else ""
        
        return original_code, mutant_code


class MutationReporter:
    """Provides reporting capabilities for mutation test results."""
    
    def __init__(self, test_suite: MutationTestSuite):
        """
        Initialize the reporter.
        
        Args:
            test_suite: MutationTestSuite to report on
        """
        self.test_suite = test_suite
    
    def summary(self) -> str:
        """
        Generate a summary report.
        
        Returns:
            Summary string
        """
        summary_lines = [
            f"Test Suite: {self.test_suite.name}",
            f"Total Mutations: {self.test_suite.total_mutations}",
            f"Killed Mutations: {self.test_suite.killed_mutations}",
            f"Survived Mutations: {self.test_suite.survived_mutations}",
            f"Mutation Score: {self.test_suite.mutation_score}%"
        ]
        
        return "\n".join(summary_lines)
    
    def write_html_report(self, output_file: str, template: Optional[str] = None) -> None:
        """
        Write an HTML report to file.
        
        Args:
            output_file: Path to output HTML file
            template: Optional custom HTML template
        """
        if template:
            html_content = self._render_custom_template(template)
        else:
            html_content = self._generate_default_html()
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def get_survived_mutations(self) -> List[MutationResult]:
        """Get list of survived mutations."""
        return [result for result in self.test_suite.results if result.is_survived]
    
    def get_killed_mutations(self) -> List[MutationResult]:
        """Get list of killed mutations."""
        return [result for result in self.test_suite.results if result.is_killed]
    
    def _generate_default_html(self) -> str:
        """Generate default HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mutation Testing Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }
                .mutation-result { margin: 10px 0; padding: 10px; border-left: 4px solid; }
                .killed { border-left-color: #28a745; background-color: #d4edda; }
                .survived { border-left-color: #dc3545; background-color: #f8d7da; }
                .code { font-family: monospace; background-color: #f8f9fa; padding: 5px; }
            </style>
        </head>
        <body>
            <h1>Mutation Testing Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Test Suite:</strong> {suite_name}</p>
                <p><strong>Total Mutations:</strong> {total_mutations}</p>
                <p><strong>Killed Mutations:</strong> {killed_mutations}</p>
                <p><strong>Survived Mutations:</strong> {survived_mutations}</p>
                <p><strong>Mutation Score:</strong> {mutation_score}%</p>
            </div>
            
            <div class="details">
                <h2>Mutation Details</h2>
                {mutation_details}
            </div>
        </body>
        </html>
        """
        
        mutation_details = ""
        for result in self.test_suite.results:
            status_class = "killed" if result.is_killed else "survived"
            mutation_details += f"""
            <div class="mutation-result {status_class}">
                <h3>{result.id} - {result.status.upper()}</h3>
                <p><strong>File:</strong> {result.filename}:{result.line_number}</p>
                <p><strong>Original:</strong> <span class="code">{result.original_code}</span></p>
                <p><strong>Mutant:</strong> <span class="code">{result.mutant_code}</span></p>
                {f'<p><strong>Killed by:</strong> {result.test_that_killed}</p>' if result.test_that_killed else ''}
            </div>
            """
        
        return html_template.format(
            suite_name=self.test_suite.name,
            total_mutations=self.test_suite.total_mutations,
            killed_mutations=self.test_suite.killed_mutations,
            survived_mutations=self.test_suite.survived_mutations,
            mutation_score=self.test_suite.mutation_score,
            mutation_details=mutation_details
        )
    
    def _render_custom_template(self, template: str) -> str:
        """Render custom template with suite data."""
        return template.replace("{{ suite_name }}", self.test_suite.name).replace(
            "{{ mutation_score }}", str(self.test_suite.mutation_score)
        )
