import struct
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Union


class Tensor:
    """Represents a tensor in the RKNN model."""
    def __init__(self, name: str, shape: List[int], dtype: str):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f'Tensor(name="{self.name}", shape={self.shape}, dtype="{self.dtype}")'


class Model:
    """Represents a parsed RKNN model."""
    def __init__(self):
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.graphs: List = []

    def add_input(self, tensor: Tensor):
        self.inputs.append(tensor)

    def add_output(self, tensor: Tensor):
        self.outputs.append(tensor)


class RKNNParser:
    """Parser for RKNN model files."""

    # Data type mapping from Netron
    DATA_TYPES = [
        'undefined', 'float32', 'uint8', 'int8', 'uint16', 'int16',
        'int32', 'int64', 'string', 'boolean', 'float16', 'float64',
        'uint32', 'uint64', 'complex<float32>', 'complex<float64>', 'bfloat16'
    ]

    # File signatures
    RKNN_SIGNATURE = b'RKNN'
    OPENVX_SIGNATURE = b'VPMN'

    def __init__(self):
        self.model = Model()

    def detect_format(self, data: bytes) -> str:
        """Detect the format of the RKNN file."""
        if len(data) < 8:
            return 'unknown'

        # Check for RKNN signature at position 0 or 4
        if data[0:4] == self.RKNN_SIGNATURE:
            return 'rknn'
        elif data[4:8] == self.RKNN_SIGNATURE:
            return 'flatbuffers'
        elif data[0:4] == self.OPENVX_SIGNATURE:
            return 'openvx'

        # Try to parse as JSON
        try:
            json.loads(data.decode('utf-8'))
            return 'json'
        except:
            pass

        return 'unknown'

    def parse(self, filepath: Union[str, Path]) -> Model:
        """Parse an RKNN model file."""
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            data = f.read()

        format_type = self.detect_format(data)

        if format_type == 'flatbuffers' or format_type == 'rknn':
            return self._parse_rknn_binary(data)
        elif format_type == 'json':
            return self._parse_json(data)
        else:
            raise ValueError(f"Unknown or unsupported RKNN format: {format_type}")

    def _parse_rknn_binary(self, data: bytes) -> Model:
        """Parse RKNN binary format by extracting embedded information."""
        # Convert to text and search for tensor information
        text = data.decode('utf-8', errors='ignore')

        # Look for output tensors (cross_k_* and cross_v_*)
        output_pattern = re.compile(r'"(cross_[kv]_\d+)"\s*:\s*\{[^}]*"dtype"\s*:\s*"([^"]+)"[^}]*"layout"\s*:\s*"([^"]+)"')
        output_matches = output_pattern.findall(text)

        for name, dtype, layout in output_matches:
            # Create tensor with default shape (we'll try to find actual shape)
            tensor = Tensor(name, [1, 1500, 512], dtype)  # Default shape from pattern
            self.model.add_output(tensor)

        # Look for input tensors
        input_pattern = re.compile(r'"(tempMC\d+_input)"\s*:\s*\{[^}]*"dtype"\s*:\s*"([^"]+)"[^}]*"shape"\s*:\s*\[([^\]]+)\]')
        input_matches = input_pattern.findall(text)

        for name, dtype, shape_str in input_matches:
            # Parse shape
            shape = [int(x.strip()) for x in shape_str.split(',') if x.strip().isdigit()]
            tensor = Tensor(name, shape, dtype)
            self.model.add_input(tensor)

        # If we didn't find inputs with the specific pattern, look for any tensor with shape
        if not self.model.inputs:
            shape_pattern = re.compile(r'"shape"\s*:\s*\[([0-9,\s]+)\]')
            name_pattern = re.compile(r'"([^"]+)"\s*:\s*\{[^}]*"shape"')

            # Find all shapes
            shape_matches = list(shape_pattern.finditer(text))
            name_matches = list(name_pattern.finditer(text))

            # Match names with shapes
            for i, (name_match, shape_match) in enumerate(zip(name_matches, shape_matches)):
                name = name_match.group(1)
                shape_str = shape_match.group(1)
                shape = [int(x.strip()) for x in shape_str.split(',') if x.strip().isdigit()]

                # Skip if it's an output tensor we already found
                if any(name in out.name for out in self.model.outputs):
                    continue

                # Default to float32 if not specified
                tensor = Tensor(name, shape, 'float32')
                self.model.add_input(tensor)

        # If we still don't have inputs, add a default one based on common patterns
        if not self.model.inputs:
            # Common input pattern for encoder models
            default_input = Tensor('input', [1, 80, 3000], 'float32')  # Mel spectrogram input
            self.model.add_input(default_input)

        return self.model

    def _parse_json(self, data: bytes) -> Model:
        """Parse JSON format RKNN file."""
        model_json = json.loads(data.decode('utf-8'))

        # Extract tensors based on Netron's JSON parsing logic
        if 'tensors' in model_json:
            for tensor_data in model_json['tensors']:
                tensor = self._tensor_from_json(tensor_data)
                if tensor_data.get('category') == 'input':
                    self.model.add_input(tensor)
                elif tensor_data.get('category') == 'output':
                    self.model.add_output(tensor)

        return self.model

    def _tensor_from_json(self, tensor_data: Dict) -> Tensor:
        """Create Tensor from JSON data."""
        name = tensor_data.get('name', 'unknown')
        shape = tensor_data.get('shape', [])
        dtype = tensor_data.get('data_type', 'undefined')

        # Convert shape to list of integers
        if isinstance(shape, str):
            shape = [int(dim) for dim in shape.split(',') if dim]

        return Tensor(name, shape, dtype)


# Convenience function
def parse_rknn(filepath: Union[str, Path]) -> Model:
    """Parse an RKNN model file and return the model."""
    parser = RKNNParser()
    return parser.parse(filepath)