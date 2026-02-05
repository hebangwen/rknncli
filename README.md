# RKNN CLI

A command line tool for parsing and displaying RKNN model information with FlatBuffers support.

## Features

- Parse both FlatBuffers binary data and JSON metadata from RKNN files
- Extract detailed model information including format, compiler, runtime
- Display input/output tensor information with layout (NCHW) and data type
- Visualize FlatBuffers graph with Graphviz (SVG output)
- Python 3.8+ compatible

## Installation

### From PyPI (Recommended)

```bash
pip install rknncli
```

### Development Installation

```bash
git clone https://github.com/your-username/rknncli.git
cd rknncli
pip install -e .
```

## Usage

```bash
rknncli <path-to-rknn-model>
```

Example:

```bash
rknncli assets/base-encoder.rknn
```

Graph visualization:

```bash
rknncli assets/base-encoder.rknn --draw base-encoder.svg
```

Note: Graph visualization requires Graphviz installed and `dot` available in `PATH`.

## Output Format

The tool prints comprehensive model information including FlatBuffers metadata:

```
Model: rknn model
Target Platform: rk3588
Format: RKNPU v2
Source: ONNX
Compiler: 2.1.0+708089d1(compiler version: 2.1.0)
Runtime: rk3588
Number of graphs: 1

Input information
--------------------------------------------------------------------------------
  ValueInfo "base-mel": type FLOAT32, shape [1, 80, 3000], layout NCHW,

Output information
--------------------------------------------------------------------------------
  ValueInfo "cross_k_0": type FLOAT16, shape [1, 1500, 512], layout NCHW,
  ValueInfo "cross_v_0": type FLOAT16, shape [1, 1500, 512], layout NCHW,
```

## Development

### Prerequisites

- Python 3.8+
- FlatBuffers compiler (for schema updates)

### Setup

```bash
# Clone the repository
git clone https://github.com/hebangwen/rknncli.git
cd rknncli

# Install in development mode
pip install -e .

# Run tests
rknncli assets/base-encoder.rknn
```

### Updating FlatBuffers Schema

If you need to update the RKNN schema:

```bash
cd schemas
./generate_schema.sh
```

### Publishing to PyPI

To publish a new version:

```bash
# Build the package
python -m build

# Upload to PyPI
./scripts/upload_to_pypi.sh
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT License - see LICENSE file for details.
