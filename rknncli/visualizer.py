"""RKNN model visualization using Graphviz."""

from pathlib import Path
from typing import Dict, Any, List
from graphviz import Digraph


class RKNNVisualizer:
    """Visualizer for RKNN models."""

    def __init__(self):
        self.dot = Digraph(format='svg')
        self.dot.attr(rankdir='TB', size='12,8')
        self.dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

    def add_nodes(self, nodes: List[Dict[str, Any]]):
        """Add nodes to the graph."""
        for node in nodes:
            # Color nodes based on type
            color = self._get_node_color(node.get('type', 'Unknown'))
            self.dot.node(
                node['id'],
                node['label'],
                fillcolor=color
            )

    def add_edges(self, edges: List[Dict[str, Any]]):
        """Add edges to the graph."""
        for edge in edges:
            self.dot.edge(edge['from'], edge['to'])

    def _get_node_color(self, node_type: str) -> str:
        """Get color for node based on type."""
        color_map = {
            'Conv2D': 'lightblue',
            'Relu': 'lightgreen',
            'MaxPool': 'orange',
            'Add': 'yellow',
            'Mul': 'pink',
            'MatMul': 'cyan',
            'Softmax': 'red',
            'Reshape': 'gray',
            'Input': 'darkgreen',
            'Output': 'darkred'
        }
        return color_map.get(node_type, 'white')

    def render(self, output_path: Path) -> Path:
        """Render the graph to SVG file."""
        output_file = self.dot.render(str(output_path), cleanup=True)
        return Path(output_file)

    def create_graph_from_info(self, graph_info: Dict[str, Any]):
        """Create graph from parsed model info."""
        # Add input node
        self.dot.node('input', 'Input', fillcolor='darkgreen', shape='ellipse')

        # Add operation nodes
        self.add_nodes(graph_info.get('nodes', []))

        # Add output node
        self.dot.node('output', 'Output', fillcolor='darkred', shape='ellipse')

        # Create edges (simplified - connect sequentially)
        nodes = graph_info.get('nodes', [])
        if nodes:
            # Connect input to first node
            self.dot.edge('input', nodes[0]['id'])

            # Connect nodes sequentially
            for i in range(len(nodes) - 1):
                self.dot.edge(nodes[i]['id'], nodes[i + 1]['id'])

            # Connect last node to output
            self.dot.edge(nodes[-1]['id'], 'output')

    def save_svg(self, output_path: Path, graph_info: Dict[str, Any]) -> Path:
        """Create and save visualization as SVG."""
        self.create_graph_from_info(graph_info)
        return self.render(output_path)