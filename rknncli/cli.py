"""RKNN CLI - Command line interface for parsing RKNN models."""

import argparse
import sys
from pathlib import Path

from rknncli.parser import RKNNParser


def format_shape(size: list) -> list:
    """Format tensor shape with dimension names.

    Args:
        size: List of dimension sizes.

    Returns:
        List of dimension names (strings) or dimension values.
    """
    return size


def get_dtype_str(dtype_info: dict) -> str:
    """Get data type string from dtype info.

    Args:
        dtype_info: Dictionary containing dtype information.

    Returns:
        String representation of the data type.
    """
    if isinstance(dtype_info, dict):
        # Try vx_type first, then qnt_type
        vx_type = dtype_info.get("vx_type", "").strip()
        if vx_type:
            return vx_type.upper()
        qnt_type = dtype_info.get("qnt_type", "").strip()
        if qnt_type:
            return qnt_type.upper()
    return "FLOAT"


def print_input_info(parser: RKNNParser) -> None:
    """Print input tensor information.

    Args:
        parser: RKNNParser instance.
    """
    print("Input information")
    print("-" * 80)

    inputs = parser.get_input_info()
    for tensor in inputs:
        name = tensor.get("url", f"tensor_{tensor.get('tensor_id', 0)}")
        dtype_info = tensor.get("dtype", {})
        dtype = get_dtype_str(dtype_info)
        size = tensor.get("size", [])
        shape = format_shape(size)

        shape_str = "[" + ", ".join(f"'{s}'" if isinstance(s, str) else str(s) for s in shape) + "]"
        print(f'  ValueInfo "{name}": type {dtype}, shape {shape_str},')


def print_output_info(parser: RKNNParser) -> None:
    """Print output tensor information.

    Args:
        parser: RKNNParser instance.
    """
    print("Output information")
    print("-" * 80)

    outputs = parser.get_output_info()
    for tensor in outputs:
        name = tensor.get("url", f"tensor_{tensor.get('tensor_id', 0)}")
        dtype_info = tensor.get("dtype", {})
        dtype = get_dtype_str(dtype_info)
        size = tensor.get("size", [])
        shape = format_shape(size)

        shape_str = "[" + ", ".join(f"'{s}'" if isinstance(s, str) else str(s) for s in shape) + "]"
        print(f'  ValueInfo "{name}": type {dtype}, shape {shape_str},')


def print_model_summary(parser: RKNNParser) -> None:
    """Print model summary information.

    Args:
        parser: RKNNParser instance.
    """
    print(f"Model: {parser.get_model_name()}")
    print(f"Version: {parser.get_version()}")
    platforms = parser.get_target_platform()
    if platforms:
        print(f"Target Platform: {', '.join(platforms)}")
    print()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        prog="rknncli",
        description="A command line tool for parsing and displaying RKNN model information.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to the RKNN model file",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: File not found: {model_path}", file=sys.stderr)
        return 1

    if not model_path.is_file():
        print(f"Error: Not a file: {model_path}", file=sys.stderr)
        return 1

    try:
        rknn_parser = RKNNParser(model_path)
    except ValueError as e:
        print(f"Error: Failed to parse RKNN file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print model summary
    print_model_summary(rknn_parser)

    # Print input information
    print_input_info(rknn_parser)
    print()

    # Print output information
    print_output_info(rknn_parser)

    return 0


if __name__ == "__main__":
    sys.exit(main())
