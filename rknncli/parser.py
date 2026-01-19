"""RKNN file parser using FlatBuffers."""

import struct
from pathlib import Path
from typing import Dict, List, Any, Optional


class RKNNParser:
    """Parser for RKNN model files."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data = None
        self._parse_header()

    def _parse_header(self):
        """Parse RKNN file header."""
        with open(self.file_path, 'rb') as f:
            self.data = f.read()

        # RKNN files start with file identifier
        if self.data[:4] != b'RKNN':
            raise ValueError("Not a valid RKNN file")

    def get_model_info(self) -> Dict[str, Any]:
        """Extract basic model information."""
        # Basic info from file size and header
        info = {
            'file_size': len(self.data),
            'file_identifier': self.data[:4].decode('ascii'),
        }

        # Try to extract more info from the binary
        # This is a simplified extraction - real RKNN parsing would be more complex
        try:
            # Look for string patterns that might indicate model info
            strings = []
            i = 0
            while i < len(self.data) - 4:
                # Check for reasonable string length
                if self.data[i] == 0 and i + 4 < len(self.data):
                    # Look for string length prefix
                    length = struct.unpack('<I', self.data[i+1:i+5])[0]
                    if 0 < length < 1000 and i + 5 + length < len(self.data):
                        string_data = self.data[i+5:i+5+length]
                        if all(32 <= b <= 126 for b in string_data):  # Printable ASCII
                            strings.append(string_data.decode('ascii', errors='ignore'))
                            i += 5 + length
                            continue
                i += 1

            # Filter out likely model metadata
            info['strings'] = [s for s in strings if len(s) > 3 and not s.isdigit()][:10]

        except Exception as e:
            info['parse_error'] = str(e)

        return info

    def get_io_info(self) -> Dict[str, Any]:
        """Extract input/output information."""
        io_info = {
            'inputs': [],
            'outputs': [],
            'tensors': []
        }

        # Since we don't have the full schema compiled, we'll do basic binary analysis
        # Look for tensor-like structures in the binary
        try:
            # Search for common tensor shape patterns [batch, height, width, channels]
            for i in range(0, len(self.data) - 16, 4):
                # Look for 4 consecutive integers that could be dimensions
                dims = struct.unpack('<IIII', self.data[i:i+16])
                # Reasonable dimension checks
                if all(0 < d <= 10000 for d in dims):
                    io_info['tensors'].append({
                        'shape': list(dims),
                        'offset': i
                    })
                    if len(io_info['tensors']) > 20:  # Limit results
                        break
        except:
            pass

        return io_info

    def get_graph_info(self) -> Dict[str, Any]:
        """Extract graph structure for visualization."""
        graph_info = {
            'nodes': [],
            'edges': []
        }

        # Basic node detection from binary patterns
        # This is a heuristic approach
        try:
            # Look for operation type strings
            op_types = ['Conv2D', 'Relu', 'MaxPool', 'Add', 'Mul', 'MatMul', 'Softmax', 'Reshape']
            node_id = 0

            for op in op_types:
                op_bytes = op.encode('ascii')
                offset = 0
                while True:
                    pos = self.data.find(op_bytes, offset)
                    if pos == -1:
                        break

                    graph_info['nodes'].append({
                        'id': f'node_{node_id}',
                        'type': op,
                        'label': f'{op}_{node_id}',
                        'offset': pos
                    })
                    node_id += 1
                    offset = pos + 1
        except:
            pass

        return graph_info