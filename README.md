# RKNN CLI Tool

A command-line tool for analyzing and visualizing RKNN (Rockchip Neural Network) model files.

## Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e .
```

## Usage

### Show Model I/O Information

```bash
rknncli -io model.rknn
```

This displays:
- File size
- Model metadata
- Input/output tensor information
- Detected operation strings in the model

### Generate Visualization

```bash
rknncli --draw model.rknn
```

This creates an SVG visualization of the model graph showing:
- Input/output nodes
- Operation nodes (color-coded by type)
- Connections between operations

You can specify a custom output path:
```bash
rknncli --draw model.rknn -o visualization.svg
```

## Example Output

```bash
$ rknncli -io assets/base-encoder.rknn
Model: base-encoder.rknn
File size: 54,492,267 bytes

Input/Output Information:
{
  "inputs": [],
  "outputs": [],
  "tensors": [...]
}

$ rknncli --draw assets/base-encoder.rknn
Visualization saved to: base-encoder.svg
Found 863 nodes in the model
Node type summary:
  - Add: 167
  - Mul: 316
  - MatMul: 276
  - Reshape: 104
```

## Development

The project structure:
```
rknncli/
├── rknncli/
│   ├── __init__.py
│   ├── cli.py          # Command-line interface
│   ├── parser.py       # RKNN file parser
│   └── visualizer.py   # Graphviz visualization
├── tests/              # Test directory
├── src/
│   └── rknn.fbs       # FlatBuffers schema (reference)
└── assets/
    └── base-encoder.rknn  # Example model
```

## Dependencies

- flatbuffers: For parsing RKNN binary format
- graphviz: For creating visualizations
- click: For command-line interface