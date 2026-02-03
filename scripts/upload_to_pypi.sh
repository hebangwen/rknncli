#!/bin/bash
# Script to upload rknncli package to PyPI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if twine is installed
if ! command -v twine > /dev/null 2>&1; then
    print_error "twine is not installed!"
    echo "Please install twine: pip install twine"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

# Check if distributions exist
if [[ ! -d "dist" ]]; then
    print_error "dist/ directory not found!"
    echo "Please run 'python -m build' first to build the package."
    exit 1
fi

# Find package files
wheel_files=(dist/*.whl)
tarball_files=(dist/*.tar.gz)

if [[ ${#wheel_files[@]} -eq 0 ]]; then
    print_error "No wheel files found in dist/ directory!"
    exit 1
fi

print_info "Found packages to upload:"
for pkg in dist/*; do
    echo "  - $(basename "$pkg")"
done

# Check PyPI configuration
if [[ ! -f "$HOME/.pypirc" ]]; then
    print_error "PyPI configuration file not found at $HOME/.pypirc"
    echo "Please configure your PyPI credentials in ~/.pypirc"
    exit 1
fi

# Upload to PyPI
print_info "Uploading to PyPI..."
twine upload dist/*

if [[ $? -eq 0 ]]; then
    print_info "Successfully uploaded to PyPI!"
    echo ""
    echo "Your package is now available at:"
    echo "https://pypi.org/project/rknncli/"
else
    print_error "Upload failed!"
    exit 1
fi

echo ""
print_info "Upload completed successfully!"
print_info "You can now install the package with: pip install rknncli"},