# RKNN CLI

A command line tool for parsing and displaying RKNN model information.

## Installation

```bash
pip install -e .
```

## Usage

```bash
rknncli <path-to-rknn-model>
```

Example:

```bash
rknncli assets/yolov5s-640-640.rknn
```

## Output Format

The tool prints model information in the following format:

```
Model: rknn model
Version: 1.6.2-source_code
Target Platform: rk3588

Input information
--------------------------------------------------------------------------------
  ValueInfo "images": type INT8, shape ['batch', 3, 'height', 'width'],

Output information
--------------------------------------------------------------------------------
  ValueInfo "output0": type INT8, shape ['batch', 255, 80, 80],
  ValueInfo "286": type INT8, shape ['batch', 255, 40, 40],
  ValueInfo "288": type INT8, shape ['batch', 255, 20, 20'],
```

## Development

```bash
# Install in development mode
pip install -e .

# Run the CLI
python -m rknncli.cli assets/yolov5s-640-640.rknn
```
