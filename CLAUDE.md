# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python project repository for working with RKNN (Rockchip Neural Network) files. The repository contains:
- A CLI tool (`rknncli`) for analyzing and visualizing RKNN model files
- A FlatBuffers schema file (`src/rknn.fbs`) defining the RKNN model format
- RKNN model files in the `assets/` directory (currently `base-encoder.rknn`)
- A Python virtual environment (`.venv/`)

## Project Structure

```
/Users/hebangwen/Code/rknncli/
├── .venv/              # Python virtual environment
├── assets/             # RKNN model files
│   └── base-encoder.rknn
├── rknncli/            # Python package
│   ├── __init__.py
│   ├── cli.py         # Command-line interface
│   ├── parser.py      # RKNN file parser
│   └── visualizer.py  # Graphviz visualization
├── src/                # Source code
│   └── rknn.fbs       # FlatBuffers schema for RKNN format
├── pyproject.toml     # Project configuration
├── README.md          # Documentation
└── .gitignore         # Git ignore rules
```

## Development Setup

The repository uses a Python virtual environment. To activate it:
```bash
source .venv/bin/activate
```

Install the package in development mode:
```bash
pip install -e .
```

## Common Commands

### Analyze RKNN Model I/O
```bash
rknncli -io assets/model.rknn
```

### Generate Model Visualization
```bash
rknncli --draw assets/model.rknn
rknncli --draw assets/model.rknn -o custom_name.svg
```

## Key Technical Details

- **FlatBuffers Schema**: The `src/rknn.fbs` file defines the structure of RKNN model files using FlatBuffers serialization format
- **RKNN Files**: Neural network model files in Rockchip's format (note: these are ignored by git as per `.gitignore:218`)
- **Dependencies**: flatbuffers, graphviz, click

## Working with RKNN Files

When working with RKNN model files:
1. They should be placed in the `assets/` directory
2. They will be automatically ignored by git (see `.gitignore:218`)
3. The FlatBuffers schema in `src/rknn.fbs` defines their structure
4. Use `rknncli` tool to analyze or visualize the models