"""Command-line interface for rknncli."""

import json
from pathlib import Path

import click

from .parser import RKNNParser
from .visualizer import RKNNVisualizer


@click.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('-io', '--info', 'show_io', is_flag=True, help='Show model input/output information')
@click.option('--draw', 'draw_graph', is_flag=True, help='Draw model graph as SVG')
@click.option('-o', '--output', type=click.Path(path_type=Path), help='Output path for SVG file (default: model_name.svg)')
def main(model_path: Path, show_io: bool, draw_graph: bool, output: Path):
    """RKNN CLI tool for analyzing and visualizing RKNN model files.

    Examples:
        rknncli -io model.rknn          # Show I/O information
        rknncli --draw model.rknn       # Generate visualization
        rknncli --draw model.rknn -o viz.svg  # Custom output path
    """
    if not show_io and not draw_graph:
        click.echo("Error: Please specify either -io or --draw option", err=True)
        click.echo("Use --help for usage information")
        return 1

    try:
        # Parse the RKNN file
        parser = RKNNParser(model_path)

        if show_io:
            # Show I/O information
            io_info = parser.get_io_info()
            model_info = parser.get_model_info()

            click.echo(f"Model: {model_path.name}")
            click.echo(f"File size: {model_info['file_size']:,} bytes")
            click.echo("\nInput/Output Information:")
            click.echo(json.dumps(io_info, indent=2))

            if model_info.get('strings'):
                click.echo("\nDetected metadata strings:")
                for s in model_info['strings']:
                    click.echo(f"  - {s}")

        if draw_graph:
            # Generate visualization
            graph_info = parser.get_graph_info()

            if not output:
                output = model_path.with_suffix('.svg')

            visualizer = RKNNVisualizer()
            svg_path = visualizer.save_svg(output.with_suffix(''), graph_info)

            click.echo(f"\nVisualization saved to: {svg_path}")
            click.echo(f"Found {len(graph_info['nodes'])} nodes in the model")

            # Show node types if verbose
            if graph_info['nodes']:
                node_types = {}
                for node in graph_info['nodes']:
                    node_type = node.get('type', 'Unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1

                click.echo("\nNode type summary:")
                for node_type, count in node_types.items():
                    click.echo(f"  - {node_type}: {count}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())