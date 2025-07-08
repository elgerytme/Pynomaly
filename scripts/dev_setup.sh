#!/bin/bash

# Development environment setup script
# This script wraps the two main commands for setting up the development environment

set -e

echo "Setting up Pynomaly development environment..."

# Create the dev environment
echo "Creating development environment..."
hatch env create dev

# Run the development setup
echo "Installing Pynomaly in editable mode and setting up pre-commit hooks..."
hatch run dev:setup

echo "Development environment setup complete!"
echo "You can now use 'hatch run dev:' to execute commands in the development environment."
