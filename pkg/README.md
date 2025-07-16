# Third-Party Packages Directory

This directory contains third-party packages that need to be included in the repository.

## Directory Structure

- `vendor_dependencies/` - Vendored third-party dependencies
- `custom_forks/` - Customized forks of open-source packages

## Guidelines

- Only include packages that cannot be managed through standard package managers
- Each package should have its own subdirectory with:
  - README.md explaining why it's included
  - LICENSE file from original package
  - CHANGELOG.md for any local modifications
  - Original source attribution

## Adding New Packages

1. Create a subdirectory under the appropriate category
2. Include all required files (README, LICENSE, etc.)
3. Update this README with package information
4. Add to build configuration if needed

## Current Packages

None currently. This structure is prepared for future third-party package needs.