# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-02-03

### Added
- FlatBuffers schema support for parsing binary RKNN model data
- Merged model information display combining FlatBuffers and JSON data
- Layout information (NCHW) extracted from FlatBuffers Generator field
- Data type information from FlatBuffers quant_tab (float32/float16)
- generate_schema.sh script for regenerating Python code from rknn.fbs
- Complete documentation for schema generation in schemas/README.md

### Changed
- Default behavior now parses both FlatBuffers and JSON data
- Removed --fb and --fb-only command line options
- Optimized output format: Layout info on same line as tensor info
- Version information consolidated into compiler field display
- Updated Python compatibility to 3.8+

### Fixed
- Python 3.8 compatibility issues with type annotations
- Import path fixes for FlatBuffers generated modules

## [0.1.0] - 2024-01-XX

### Added
- Initial release with basic RKNN model parsing
- JSON model information extraction
- Input/Output tensor information display
- Command line interface with basic options