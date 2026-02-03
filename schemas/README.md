# FlatBuffers Schema Generation

This directory contains the FlatBuffers schema for RKNN models and a script to generate Python parsing code.

## Files

- `rknn.fbs` - FlatBuffers schema definition for RKNN model format
- `generate_schema.sh` - Shell script to generate Python code from the schema

## Usage

### Prerequisites

Install the FlatBuffers compiler (`flatc`):

```bash
# Ubuntu/Debian
sudo apt-get install flatbuffers-compiler

# macOS
brew install flatbuffers

# From source
# See: https://github.com/google/flatbuffers
```

### Generate Python Code

Run the generation script:

```bash
./generate_schema.sh
```

This will:
1. Generate Python classes from the FlatBuffers schema
2. Fix import paths to match the project structure
3. Verify the generated code can be imported

### Options

```bash
# Use a different schema file
./generate_schema.sh --schema /path/to/custom.rknn.fbs

# Output to a different directory
./generate_schema.sh --output /path/to/output

# Show help
./generate_schema.sh --help
```

## Schema Structure

The `rknn.fbs` schema defines the following main components:

- **Model** - Root type containing model metadata and graphs
- **Graph** - Contains tensors, nodes, inputs and outputs
- **Tensor** - Defines tensor properties (shape, data type, etc.)
- **Node** - Represents operations in the graph
- **Type1/Type2/Type3** - Auxiliary types

## Generated Files

The script generates the following Python modules in `rknncli/schema/rknn/`:

- `Model.py` - Model class definition
- `Graph.py` - Graph class definition
- `Tensor.py` - Tensor class definition
- `Node.py` - Node class definition
- `Type1.py`, `Type2.py`, `Type3.py` - Auxiliary type definitions
- `__init__.py` - Package initialization

## Updating the Schema

If you need to update the RKNN format:

1. Modify `rknn.fbs` according to the FlatBuffers schema syntax
2. Run `./generate_schema.sh` to regenerate the Python code
3. Update the parser code in `rknncli/parser.py` if needed

## Troubleshooting

- **flatc not found**: Install FlatBuffers compiler
- **Import errors**: The script automatically fixes import paths, but if issues persist, check the generated files
- **Generation fails**: Check the schema file for syntax errors using `flatc --jsonschema rknn.fbs`