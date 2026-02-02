import os
import subprocess
from pathlib import Path
from typing import Optional, Union

import graphviz

from .parser import Model, Tensor


class ModelVisualizer:
    """Generate Graphviz visualizations of RKNN models."""

    def __init__(self, model: Model):
        self.model = model

    def to_dot(self) -> str:
        """Convert the model to DOT format."""
        dot = graphviz.Digraph(comment='RKNN Model')
        dot.attr(rankdir='TB', bgcolor='white')
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

        # Create subgraph for inputs
        with dot.subgraph(name='cluster_inputs') as c:
            c.attr(label='Inputs', style='rounded,filled', fillcolor='lightblue', fontname='Arial Bold')
            for i, tensor in enumerate(self.model.inputs):
                node_id = f'input_{i}'
                label = self._format_tensor_label(tensor)
                c.node(node_id, label, fillcolor='lightcyan')

        # Create subgraph for outputs
        with dot.subgraph(name='cluster_outputs') as c:
            c.attr(label='Outputs', style='rounded,filled', fillcolor='lightgreen', fontname='Arial Bold')
            for i, tensor in enumerate(self.model.outputs):
                node_id = f'output_{i}'
                label = self._format_tensor_label(tensor)
                c.node(node_id, label, fillcolor='palegreen')

        # Connect inputs to outputs (simplified - real implementation would trace through graph)
        for i, input_tensor in enumerate(self.model.inputs):
            for j, output_tensor in enumerate(self.model.outputs):
                dot.edge(f'input_{i}', f'output_{j}', style='dashed', color='gray')

        return dot.source

    def _format_tensor_label(self, tensor: Tensor) -> str:
        """Format a tensor as a DOT node label."""
        shape_str = ' × '.join(str(dim) for dim in tensor.shape) if tensor.shape else 'scalar'
        return f"{tensor.name}\\n{shape_str}\\n{tensor.dtype}"

    def save_svg(self, output_path: Union[str, Path], format: str = 'svg') -> Path:
        """Save the visualization as an SVG file."""
        output_path = Path(output_path)

        # Create the graph
        dot = self.to_dot()

        # Render to file
        try:
            # Use graphviz to render
            gv = graphviz.Source(dot)
            rendered_path = gv.render(filename=output_path.stem, directory=output_path.parent, format=format, cleanup=True)
            return Path(rendered_path)
        except Exception as e:
            # Fallback: try to use dot command directly
            try:
                dot_file = output_path.with_suffix('.dot')
                with open(dot_file, 'w') as f:
                    f.write(dot)

                result = subprocess.run(
                    ['dot', '-T' + format, str(dot_file), '-o', str(output_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Clean up dot file
                if dot_file.exists():
                    dot_file.unlink()

                return output_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(f"Failed to generate visualization: {e}")


def visualize_model(model: Model, output_path: Union[str, Path], format: str = 'svg') -> Path:
    """Convenience function to visualize a model."""
    visualizer = ModelVisualizer(model)
    return visualizer.save_svg(output_path, format)