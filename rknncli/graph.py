"""Graph utilities for RKNN FlatBuffers visualization and shape inference."""

from dataclasses import dataclass
import io
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from graphviz import Digraph

from rknncli import metadata

from rknncli.schema.rknn.Graph import Graph as FBGraph


@dataclass
class TensorInfo:
    """Tensor metadata used for graph edges."""

    index: int
    name: str
    shape: Tuple[int, ...]


@dataclass
class NodeInfo:
    """Operator node metadata used for graph nodes."""

    index: int
    name: str
    op_type: str
    inputs: List[int]
    outputs: List[int]


class Graph:
    """Compute graph representation with shape inference and visualization."""

    def __init__(self, nodes: List[NodeInfo], tensors: Dict[int, TensorInfo]):
        self.nodes = nodes
        self.tensors = tensors
        self._producers: Dict[int, int] = {}
        self._inferred_shapes: Dict[int, Tuple[int, ...]] = {}
        self._build_producers()

    @classmethod
    def from_flatbuffers(cls, graph: FBGraph) -> "Graph":
        """Create a Graph from a FlatBuffers Graph."""
        tensors: Dict[int, TensorInfo] = {}
        for i in range(graph.TensorsLength()):
            tensor = graph.Tensors(i)
            name = f"tensor_{i}"
            if tensor and tensor.Name():
                name = tensor.Name().decode("utf-8")
            shape = tuple(tensor.Shape(k) for k in range(tensor.ShapeLength())) if tensor else ()
            tensors[i] = TensorInfo(index=i, name=name, shape=shape)

        nodes: List[NodeInfo] = []
        for i in range(graph.NodesLength()):
            node = graph.Nodes(i)
            if not node:
                continue
            op_type = node.Type().decode("utf-8") if node.Type() else "Unknown"
            name = node.Name().decode("utf-8") if node.Name() else ""
            inputs = [node.Inputs(j) for j in range(node.InputsLength())]
            outputs = [node.Outputs(j) for j in range(node.OutputsLength())]
            nodes.append(
                NodeInfo(
                    index=i,
                    name=name,
                    op_type=op_type,
                    inputs=inputs,
                    outputs=outputs,
                )
            )

        return cls(nodes=nodes, tensors=tensors)

    @classmethod
    def from_vpmn(cls, buffer: bytes) -> "Graph":
        """Create a Graph from a VPMN (OpenVX) buffer."""
        reader = _VPMNReader(buffer)
        nodes = reader.read_nodes()
        node_infos: List[NodeInfo] = []
        for i, node in enumerate(nodes):
            node_infos.append(
                NodeInfo(
                    index=i,
                    name=str(node.index) if node.index is not None else "",
                    op_type=node.type,
                    inputs=[],
                    outputs=[],
                )
            )
        return cls(nodes=node_infos, tensors={})

    def infer_shapes(self) -> Dict[int, Tuple[int, ...]]:
        """Populate shape info using tensor metadata from FlatBuffers."""
        inferred = {}
        for tensor_idx, tensor in self.tensors.items():
            if tensor.shape:
                inferred[tensor_idx] = tensor.shape
        self._inferred_shapes = inferred
        return inferred

    def to_graphviz(self, use_shape_labels: bool = True) -> Digraph:
        """Build a Graphviz Digraph for this Graph."""
        dot = Digraph(comment="RKNN Graph")
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="rounded,filled")

        for node in self.nodes:
            label = self._format_node_label(node)
            node_id = f"n{node.index}"
            dot.node(
                node_id,
                label=label,
                style="rounded,filled",
                fillcolor=metadata.get_node_color(node.op_type),
            )

        for node in self.nodes:
            node_id = f"n{node.index}"
            for tensor_idx in node.inputs:
                producer = self._producers.get(tensor_idx)
                if producer is None:
                    continue
                label = None
                if use_shape_labels:
                    label = self._shape_label(tensor_idx)
                dot.edge(f"n{producer}", node_id, label=label)

        return dot

    def _build_producers(self) -> None:
        for node in self.nodes:
            for tensor_idx in node.outputs:
                self._producers[tensor_idx] = node.index

    def _format_node_label(self, node: NodeInfo) -> str:
        if node.name and node.op_type:
            return f"{node.name}\n{node.op_type}"
        return node.name or node.op_type or f"node_{node.index}"

    def _shape_label(self, tensor_idx: int) -> str:
        shape = self._inferred_shapes.get(tensor_idx, ())
        if not shape:
            tensor = self.tensors.get(tensor_idx)
            if tensor and tensor.shape:
                shape = tensor.shape
        if not shape:
            return ""
        return "[" + ", ".join(str(dim) for dim in shape) + "]"

    def render_svg(self, output_path: str) -> str:
        """Render SVG visualization to the given output path."""
        self.infer_shapes()
        dot = self.to_graphviz(use_shape_labels=True)
        dot.format = "svg"
        return dot.render(filename=output_path, cleanup=True)


class _VPMNNode:
    def __init__(self, node_type: str, index: Optional[int]):
        self.type = node_type
        self.index = index


class _VPMNReader:
    """Parser for VPMN (OpenVX) models."""

    def __init__(self, buffer: bytes):
        self._buffer = buffer
        self._stream = io.BytesIO(buffer)
        self._major = 0

    def read_nodes(self) -> List[_VPMNNode]:
        signature = self._stream.read(4)
        if signature != b"VPMN":
            raise ValueError("Invalid VPMN signature")
        self._major = self._uint16()
        # minor
        self._uint16()
        self._skip(4)
        # name
        self._string(64)
        node_count = self._uint32()
        if self._major > 3:
            self._skip(296)
        elif self._major > 1:
            self._skip(288)
        else:
            self._skip(32)
        # input/output offsets and sizes
        self._uint32()
        self._uint32()
        self._uint32()
        self._uint32()
        node_offset = self._uint32()
        self._uint32()
        self._stream.seek(node_offset)
        nodes: List[_VPMNNode] = []
        for _ in range(node_count):
            node_type = self._string(64)
            index = self._uint32()
            # c
            self._uint32()
            if self._major > 3:
                self._uint32()
            nodes.append(_VPMNNode(node_type=node_type, index=index))
        return nodes

    def _uint16(self) -> int:
        return struct.unpack("<H", self._stream.read(2))[0]

    def _uint32(self) -> int:
        return struct.unpack("<I", self._stream.read(4))[0]

    def _skip(self, length: int) -> None:
        self._stream.seek(length, io.SEEK_CUR)

    def _string(self, length: int) -> str:
        data = self._stream.read(length)
        end = data.find(b"\x00")
        if end != -1:
            data = data[:end]
        return data.decode("ascii", errors="ignore")
