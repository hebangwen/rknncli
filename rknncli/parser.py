"""RKNN file parser with FlatBuffers support."""

import json
import struct
from pathlib import Path
from typing import Any, Optional

import flatbuffers
from rknncli.schema.rknn.Model import Model
from rknncli.schema.rknn.Graph import Graph
from rknncli.schema.rknn.Tensor import Tensor
from rknncli.schema.rknn.Node import Node


class RKNNParser:
    """Parser for RKNN model files with optional FlatBuffers support."""

    HEADER_SIZE = 64
    MAGIC_NUMBER = b"RKNN"

    def __init__(self, file_path: str | Path, parse_flatbuffers: bool = False):
        """Initialize parser with RKNN file path.

        Args:
            file_path: Path to the RKNN model file.
            parse_flatbuffers: Whether to parse FlatBuffers data. Defaults to False.
        """
        self.file_path = Path(file_path)
        self.header: dict[str, Any] = {}
        self.model_info: dict[str, Any] = {}
        self.fb_model: Optional[Model] = None
        self._parse(parse_flatbuffers)

    def _parse(self, parse_flatbuffers: bool) -> None:
        """Parse the RKNN file.

        Args:
            parse_flatbuffers: Whether to parse FlatBuffers data.
        """
        with open(self.file_path, "rb") as f:
            # Read header
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
            real_header_size = self.HEADER_SIZE
            if file_format <= 1:
                # only 3 uint64 for rknn-v1
                real_header_size = 24

            # Parse FlatBuffers data if requested
            if parse_flatbuffers and file_length > 0:
                fb_offset = real_header_size
                fb_data = f.read(file_length)
                if len(fb_data) == file_length:
                    self.fb_model = Model.GetRootAs(fb_data, 0)

            # Read JSON model info
            file_size = self.file_path.stat().st_size
            json_offset = real_header_size + file_length
            f.seek(json_offset)
            json_size = struct.unpack("<Q", f.read(8))[0]
            if (
                json_size <= 0 or
                json_size > file_size - real_header_size - file_length - 8
            ):
                raise ValueError(f"Invalid JSON size: {json_size}")

            json_data = f.read(json_size)

            try:
                self.model_info = json.loads(json_data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to parse model info JSON: {e}")

    def get_flatbuffers_info(self) -> dict[str, Any]:
        """Get FlatBuffers model information.

        Returns:
            Dictionary containing FlatBuffers model data.
        """
        if not self.fb_model:
            return {}

        info = {}

        # Basic model info
        if self.fb_model.Format():
            info["format"] = self.fb_model.Format().decode('utf-8')

        if self.fb_model.Generator():
            info["generator"] = self.fb_model.Generator().decode('utf-8')

        if self.fb_model.Compiler():
            info["compiler"] = self.fb_model.Compiler().decode('utf-8')

        if self.fb_model.Runtime():
            info["runtime"] = self.fb_model.Runtime().decode('utf-8')

        if self.fb_model.Source():
            info["source"] = self.fb_model.Source().decode('utf-8')

        # Graphs info
        info["num_graphs"] = self.fb_model.GraphsLength()

        # Input/Output JSON strings
        if self.fb_model.InputJson():
            info["input_json"] = self.fb_model.InputJson().decode('utf-8')

        if self.fb_model.OutputJson():
            info["output_json"] = self.fb_model.OutputJson().decode('utf-8')

        # Parse graphs
        graphs = []
        for i in range(self.fb_model.GraphsLength()):
            graph = self.fb_model.Graphs(i)
            if graph:
                graph_info = {
                    "num_tensors": graph.TensorsLength(),
                    "num_nodes": graph.NodesLength(),
                    "num_inputs": graph.InputsLength(),
                    "num_outputs": graph.OutputsLength(),
                }

                # Parse tensors
                tensors = []
                for j in range(graph.TensorsLength()):
                    tensor = graph.Tensors(j)
                    if tensor:
                        tensor_info = {
                            "data_type": tensor.DataType(),
                            "kind": tensor.Kind(),
                            "name": tensor.Name().decode('utf-8') if tensor.Name() else "",
                            "shape": [tensor.Shape(k) for k in range(tensor.ShapeLength())],
                            "size": tensor.Size(),
                            "index": tensor.Index(),
                        }
                        tensors.append(tensor_info)
                graph_info["tensors"] = tensors

                # Parse nodes
                nodes = []
                for j in range(graph.NodesLength()):
                    node = graph.Nodes(j)
                    if node:
                        node_info = {
                            "type": node.Type().decode('utf-8') if node.Type() else "",
                            "name": node.Name().decode('utf-8') if node.Name() else "",
                            "num_inputs": node.InputsLength(),
                            "num_outputs": node.OutputsLength(),
                        }
                        nodes.append(node_info)
                graph_info["nodes"] = nodes

                graphs.append(graph_info)

        info["graphs"] = graphs

        return info

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