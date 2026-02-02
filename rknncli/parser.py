"""RKNN file parser."""

import json
import struct
from pathlib import Path
from typing import Any


class RKNNParser:
    """Parser for RKNN model files."""

    HEADER_SIZE = 72
    MAGIC_NUMBER = b"RKNN"

    def __init__(self, file_path: str | Path):
        """Initialize parser with RKNN file path.

        Args:
            file_path: Path to the RKNN model file.
        """
        self.file_path = Path(file_path)
        self.header: dict[str, Any] = {}
        self.model_info: dict[str, Any] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse the RKNN file."""
        with open(self.file_path, "rb") as f:
            # Read header (72 bytes)
            header_data = f.read(self.HEADER_SIZE)
            if len(header_data) < self.HEADER_SIZE:
                raise ValueError(f"File too small: {self.file_path}")

            # Parse header
            # Bytes 0-3: Magic number "RKNN"
            magic = header_data[0:4]
            if magic != self.MAGIC_NUMBER:
                raise ValueError(f"Invalid magic number: {magic!r}, expected {self.MAGIC_NUMBER!r}")

            # Bytes 4-7: Padding (4 bytes, zeros)
            padding = header_data[4:8]

            # Bytes 8-15: File format version (8 bytes, little-endian uint64)
            file_format = struct.unpack("<Q", header_data[8:16])[0]

            # Bytes 16-23: File length (8 bytes, little-endian uint64)
            file_length = struct.unpack("<Q", header_data[16:24])[0]

            self.header = {
                "magic": magic,
                "padding": padding,
                "file_format": file_format,
                "file_length": file_length,
            }

            # Calculate JSON offset and size
            json_offset = self.HEADER_SIZE + file_length
            file_size = self.file_path.stat().st_size
            json_size = file_size - json_offset

            if json_size <= 0:
                raise ValueError(f"Invalid JSON section size: {json_size}")

            # Read JSON model info
            f.seek(json_offset)
            json_data = f.read(json_size)

            try:
                self.model_info = json.loads(json_data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to parse model info JSON: {e}")

    def get_input_info(self) -> list[dict[str, Any]]:
        """Get model input information.

        Returns:
            List of input tensor information.
        """
        inputs = []
        norm_tensors = {t["tensor_id"]: t for t in self.model_info.get("norm_tensor", [])}

        # Find input tensors from graph connections
        for conn in self.model_info.get("graph", []):
            if conn.get("left") == "input":
                tensor_id = conn.get("right_tensor_id")
                if tensor_id in norm_tensors:
                    inputs.append(norm_tensors[tensor_id])

        return inputs

    def get_output_info(self) -> list[dict[str, Any]]:
        """Get model output information.

        Returns:
            List of output tensor information.
        """
        outputs = []
        norm_tensors = {t["tensor_id"]: t for t in self.model_info.get("norm_tensor", [])}

        # Find output tensors from graph connections
        for conn in self.model_info.get("graph", []):
            if conn.get("left") == "output":
                tensor_id = conn.get("right_tensor_id")
                if tensor_id in norm_tensors:
                    outputs.append(norm_tensors[tensor_id])

        return outputs

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_info.get("name", "Unknown")

    def get_version(self) -> str:
        """Get model version."""
        return self.model_info.get("version", "Unknown")

    def get_target_platform(self) -> list[str]:
        """Get target platform."""
        return self.model_info.get("target_platform", [])
