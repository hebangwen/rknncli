import sys
from pathlib import Path

import click

from .parser import parse_rknn, Tensor
from .visualizer import visualize_model


def format_tensor_info(tensor: Tensor) -> str:
    """Format tensor information for display."""
    shape_str = ', '.join(str(dim) for dim in tensor.shape)
    return f'  ValueInfo "{tensor.name}": type {tensor.dtype}, shape [{shape_str}]'


@click.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
@click.option('-io', '--input-output', 'show_io', is_flag=True, help='Print input/output tensor information')
@click.option('--draw', 'draw_graph', is_flag=True, help='Generate model visualization')
@click.option('-o', '--output', type=click.Path(path_type=Path), help='Output path for visualization (default: model_name.svg)')
def main(model_path: Path, show_io: bool, draw_graph: bool, output: Path):
    """RKNN CLI tool for analyzing and visualizing RKNN model files."""
    try:
        # Parse the model
        model = parse_rknn(model_path)

        # Handle input/output display
        if show_io:
            print("Input information")
            print("--------------------------------------------------------------------------------")
            if model.inputs:
                for tensor in model.inputs:
                    print(format_tensor_info(tensor))
            else:
                print("  No inputs found")

            print("\nOutput information")
            print("--------------------------------------------------------------------------------")
            if model.outputs:
                for tensor in model.outputs:
                    print(format_tensor_info(tensor))
            else:
                print("  No outputs found")

        # Handle visualization
        if draw_graph:
            if output is None:
                output = model_path.with_suffix('.svg')

            print(f"\nGenerating visualization: {output}")
            try:
                visualize_model(model, output)
                print(f"Visualization saved to: {output}")
            except Exception as e:
                print(f"Error generating visualization: {e}", file=sys.stderr)
                sys.exit(1)

        # If no specific action requested, show help
        if not show_io and not draw_graph:
            click.echo(main.get_help(click.Context(main)))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()