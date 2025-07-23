# Documentation Cross-Reference Index

This file contains the cross-reference mapping between documentation files to ensure comprehensive navigation and discoverability.

## Cross-Reference Map

### Getting Started Journey
- `getting-started/index.md` → All learning paths, algorithm selection, examples
- `getting-started/first-detection.md` → Algorithm guide, examples, deployment
- `getting-started/examples.md` → All advanced guides based on use case

### Core Concepts
- `algorithms.md` → Getting started, examples, ensemble, performance
- `api.md` → Getting started, CLI, integration, examples
- `cli.md` → API, deployment, integration, examples

### Advanced Topics
- `ensemble.md` → Algorithms, examples, performance, explainability
- `explainability.md` → Algorithms, model management, examples
- `model_management.md` → Algorithms, ensemble, deployment, performance
- `performance.md` → Algorithms, deployment, streaming, troubleshooting
- `streaming.md` → Algorithms, performance, deployment, integration

### Production & Operations
- `deployment.md` → Security, performance, streaming, integration
- `integration.md` → API, deployment, streaming, security
- `security.md` → Deployment, integration, troubleshooting
- `troubleshooting.md` → All guides for problem resolution

### Reference Materials
- `architecture.md` → All implementation guides
- `configuration.md` → Deployment, security, performance
- `installation.md` → Getting started, troubleshooting

## Contextual Navigation Patterns

### By User Role
- **Beginners**: getting-started → algorithms → examples → troubleshooting
- **Data Scientists**: algorithms → model_management → ensemble → explainability → performance
- **Engineers**: deployment → integration → streaming → security → troubleshooting

### By Use Case
- **Fraud Detection**: examples → algorithms → ensemble → deployment → security
- **Network Security**: examples → streaming → performance → integration → troubleshooting
- **Quality Control**: examples → algorithms → model_management → deployment
- **IoT Monitoring**: streaming → integration → performance → deployment

### By Implementation Phase
- **Planning**: getting-started → algorithms → examples
- **Development**: api → model_management → ensemble → explainability
- **Testing**: performance → troubleshooting
- **Deployment**: deployment → integration → security → streaming
- **Operations**: troubleshooting → performance → security

## Link Templates

### Standard Cross-Reference Format
```markdown
!!! info "Related Guides"
    - **[Context]?** [Description] with [Guide Name](link.md)
    - **[Use case]?** See [Guide Name](link.md) for [specific benefit]
```

### Navigation Section Format
```markdown
=== "[Category]"
    [Description] in the [Guide Name](link.md). [Benefit statement].
```

### Inline Reference Format
```markdown
For [specific topic], see the [Guide Name](link.md) section on [specific feature].
```

## Cross-Reference Quality Checklist

- [ ] Each major guide has prerequisites clearly linked
- [ ] Related advanced topics are cross-referenced
- [ ] Practical examples link to relevant theory
- [ ] Troubleshooting is accessible from all guides
- [ ] Learning paths are clearly defined with next steps
- [ ] Implementation guides reference each other appropriately