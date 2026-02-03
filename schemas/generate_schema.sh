#!/bin/bash
# Script to generate Python code from rknn.fbs FlatBuffers schema

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if flatc is installed
if ! command -v flatc &> /dev/null; then
    print_error "flatc (FlatBuffers compiler) is not installed!"
    echo "Please install FlatBuffers:"
    echo "  - Ubuntu/Debian: sudo apt-get install flatbuffers-compiler"
    echo "  - macOS: brew install flatbuffers"
    echo "  - From source: https://github.com/google/flatbuffers"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
SCHEMA_FILE="${SCRIPT_DIR}/rknn.fbs"
OUTPUT_DIR="${PROJECT_ROOT}/rknncli/schema"
SCHEMA_NAME="rknn"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--schema)
            SCHEMA_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Generate Python code from rknn.fbs FlatBuffers schema"
            echo ""
            echo "Options:"
            echo "  -s, --schema FILE    Path to rknn.fbs schema file (default: ${SCRIPT_DIR}/rknn.fbs)"
            echo "  -o, --output DIR     Output directory for generated Python files (default: ${OUTPUT_DIR})"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if schema file exists
if [[ ! -f "${SCHEMA_FILE}" ]]; then
    print_error "Schema file not found: ${SCHEMA_FILE}"
    exit 1
fi

print_info "Using schema file: ${SCHEMA_FILE}"
print_info "Output directory: ${OUTPUT_DIR}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Generate Python files
print_info "Generating Python code from FlatBuffers schema..."
flatc --python \
    --gen-object-api \
    --gen-mutable \
    -o "${OUTPUT_DIR}" \
    "${SCHEMA_FILE}"

# Check if generation was successful
if [[ $? -eq 0 ]]; then
    print_info "Successfully generated Python files!"
else
    print_error "Failed to generate Python files"
    exit 1
fi

# Fix import paths in generated files
print_info "Fixing import paths in generated files..."
SCHEMA_OUTPUT_DIR="${OUTPUT_DIR}/${SCHEMA_NAME}"
if [[ -d "${SCHEMA_OUTPUT_DIR}" ]]; then
    # Replace relative imports with absolute imports
    find "${SCHEMA_OUTPUT_DIR}" -name "*.py" -exec sed -i.bak \
        -e "s/from ${SCHEMA_NAME}\./from rknncli.schema.${SCHEMA_NAME}./g" \
        -e "s/import ${SCHEMA_NAME}\./import rknncli.schema.${SCHEMA_NAME}./g" {} \;

    # Remove backup files
    find "${SCHEMA_OUTPUT_DIR}" -name "*.bak" -delete

    print_info "Import paths fixed!"
else
    print_warning "Schema output directory not found: ${SCHEMA_OUTPUT_DIR}"
fi

# List generated files
print_info "Generated files:"
if [[ -d "${SCHEMA_OUTPUT_DIR}" ]]; then
    for file in "${SCHEMA_OUTPUT_DIR}"/*.py; do
        if [[ -f "$file" ]]; then
            echo "  - $(basename "$file")"
        fi
    done
fi

# Verify the generated files can be imported
print_info "Verifying generated files can be imported..."
cd "${PROJECT_ROOT}"
python3 -c "
try:
    from rknncli.schema.rknn.Model import Model
    print('✓ Successfully imported Model from rknncli.schema.rknn.Model')
except ImportError as e:
    print('✗ Failed to import:', e)
    exit(1)
"

if [[ $? -eq 0 ]]; then
    print_info "All imports verified successfully!"
    print_info "Generation completed successfully!"
else
    print_error "Import verification failed"
    exit 1
fi

# Optional: Generate a summary of what was generated
print_info "Generation Summary:"
echo "Schema file: ${SCHEMA_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Generated modules:"
for file in "${SCHEMA_OUTPUT_DIR}"/*.py; do
    if [[ -f "$file" ]]; then
        module_name=$(basename "$file" .py)
        echo "  - rknncli.schema.rknn.${module_name}"
    fi
done

exit 0