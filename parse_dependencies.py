import re

# Manual parsing of the dependencies from the issue list
dependencies = [
    # (child_issue, parent_issue, child_title, parent_task)
    (30, 22, "DOC-005: Security Best Practices Guide", "C-001"),
    (29, 24, "DOC-004: Performance Benchmarking Guide", "C-003"),
    (27, 17, "DOC-002: User Guide Video Tutorials", "P-001"),
    (26, 21, "DOC-001: API Documentation Completion", "P-005"),
    (23, 12, "C-002: Multi-Environment Deployment Pipeline", "I-001"),
    (20, 12, "P-004: GraphQL API Layer", "I-001"),
    (19, 13, "P-003: CLI Command Completion", "I-002"),
    (17, 11, "P-001: Advanced Analytics Dashboard", "A-003"),
    (16, 12, "I-005: Cloud Storage Adapters", "I-001"),
    (14, 12, "I-003: Message Queue Integration", "I-001"),
    (11, 7, "A-003: Model Comparison and Selection", "D-002"),
    (9, 8, "A-001: Automated Model Retraining Workflows", "D-003"),
]

print("Found dependencies:")
for child, parent, title, parent_task in dependencies:
    print(f"- Issue #{child} ({title}) depends on #{parent} ({parent_task})")

print(f"\nTotal dependencies to process: {len(dependencies)}")
