# Pynomaly Documentation

This directory contains the documentation for the Pynomaly project, built using [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## Quick Start

### Building Documentation Locally

1. **Build the static site:**
   ```bash
   hatch run docs:build
   ```
   This creates a static site in the `site/` directory.

2. **Serve with live-reload:**
   ```bash
   hatch run docs:serve
   ```
   This starts a development server at `localhost:8080` with live-reload enabled.

### Configuration

The documentation configuration is located in:
- Main config: `mkdocs.yml` (root directory)
- Source config: `config/docs/mkdocs.yml`

### Environment Setup

The documentation environment is managed through Hatch and includes:
- `mkdocs>=1.6.0`
- `mkdocs-material>=9.5.0`
- `mkdocstrings[python]>=0.27.0`
- `mkdocs-gen-files>=0.5.0`
- `mkdocs-literate-nav>=0.6.0`

## GitHub Pages Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. The deployment is handled by the `.github/workflows/deploy-docs.yml` GitHub Actions workflow.

### Deployment Process

1. **Trigger**: Push to `main` branch with changes to:
   - `docs/**` (any documentation files)
   - `mkdocs.yml` (configuration file)
   - `.github/workflows/deploy-docs.yml` (workflow file)

2. **Build Steps**:
   - Sets up Python 3.11
   - Installs Hatch
   - Creates the docs environment
   - Builds the static site using `hatch run docs:build`
   - Uploads the `site/` directory as an artifact

3. **Deploy Steps**:
   - Deploys the artifact to GitHub Pages
   - Site becomes available at the configured GitHub Pages URL

### Manual Deployment

You can also trigger the deployment manually:

1. Go to the Actions tab in your GitHub repository
2. Select the "Deploy Documentation to GitHub Pages" workflow
3. Click "Run workflow"
4. Choose the branch (usually `main`) and click "Run workflow"

## Directory Structure

```
docs/
├── README.md                           # This file
├── index.md                           # Homepage
├── getting-started/                   # Getting started guides
├── user-guides/                       # User documentation
├── developer-guides/                  # Developer documentation
├── examples/                          # Examples and tutorials
├── reference/                         # API reference
├── deployment/                        # Deployment guides
└── ...                               # Other documentation sections
```

## Contributing to Documentation

1. **Edit markdown files** in the `docs/` directory
2. **Test locally** using `hatch run docs:serve`
3. **Commit and push** to the `main` branch
4. **GitHub Pages deployment** will automatically update the live site

## Troubleshooting

### Common Issues

1. **Build fails**: Check that all referenced files exist and links are correct
2. **Serve not working**: Ensure the docs environment is created with `hatch env create docs`
3. **Broken links**: Use relative paths and ensure all referenced files exist

### Environment Issues

If you encounter environment issues:

```bash
# Remove and recreate the docs environment
hatch env remove docs
hatch env create docs

# Try building again
hatch run docs:build
```

## Advanced Configuration

The documentation supports:

- **Material Design theme** with dark/light mode toggle
- **Code syntax highlighting** with copy buttons
- **Search functionality**
- **Navigation tabs and sections**
- **API documentation** generated from docstrings
- **Mermaid diagrams** support

For more advanced configuration options, see the [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/).
