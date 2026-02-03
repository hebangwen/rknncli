"""RKNN CLI - Command line interface for parsing RKNN models."""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

from rknncli.parser import RKNNParser


def format_shape(size: List) -> List:
    """Format tensor shape with dimension names.

    Args:
        size: List of dimension sizes.

    Returns:
        List of dimension names (strings) or dimension values.
    """
    return size


def get_dtype_str(dtype_info: Dict) -> str:
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




def print_merged_model_info(parser: RKNNParser) -> None:
    """Print merged model information from both FlatBuffers and JSON.

    Args:
        parser: RKNNParser instance with FlatBuffers support.
    """
    # Get basic model info from JSON
    print(f"Model: {parser.get_model_name()}")
    print(f"Version: {parser.get_version()}")
    platforms = parser.get_target_platform()
    if platforms:
        print(f"Target Platform: {', '.join(platforms)}")

    # Get FlatBuffers info
    fb_info = parser.get_flatbuffers_info()
    if fb_info:
        print(f"Format: {fb_info.get('format', 'Unknown')}")
        print(f"Source: {fb_info.get('source', 'Unknown')}")
        print(f"Compiler: {fb_info.get('compiler', 'Unknown')}")
        print(f"Runtime: {fb_info.get('runtime', 'Unknown')}")

        if fb_info.get("num_graphs", 0) > 0:
            print(f"Number of graphs: {fb_info['num_graphs']}")

    print()


def print_merged_io_info(parser: RKNNParser) -> None:
    """Print merged input/output information from both FlatBuffers and JSON.

    Args:
        parser: RKNNParser instance with FlatBuffers support.
    """
    # Get merged IO info
    inputs, outputs = parser.get_merged_io_info()

    # Print input information
    print("Input information")
    print("-" * 80)

    for tensor in inputs:
        name = tensor.get("url", f"tensor_{tensor.get('tensor_id', 0)}")
        dtype = tensor.get("dtype", {})

        # Get dtype string
        if isinstance(dtype, dict):
            dtype_str = get_dtype_str(dtype)
        else:
            dtype_str = str(dtype).upper()

        size = tensor.get("size", [])
        shape = format_shape(size)
        shape_str = "[" + ", ".join(f"'{s}'" if isinstance(s, str) else str(s) for s in shape) + "]"

        # Print basic info
        print(f'  ValueInfo "{name}": type {dtype_str}, shape {shape_str},')

        # Print additional info from FlatBuffers
        if "layout" in tensor:
            print(f'    Layout: {tensor["layout"]} (original: {tensor.get("layout_ori", "n/a")})')

        if "quant_info" in tensor:
            quant = tensor["quant_info"]
            if quant.get("qmethod") or quant.get("qtype"):
                print(f'    Quantization: {quant["qmethod"]} {quant["qtype"]}')

    print()

    # Print output information
    print("Output information")
    print("-" * 80)

    for tensor in outputs:
        name = tensor.get("url", f"tensor_{tensor.get('tensor_id', 0)}")
        dtype = tensor.get("dtype", {})

        # Get dtype string
        if isinstance(dtype, dict):
            dtype_str = get_dtype_str(dtype)
        else:
            dtype_str = str(dtype).upper()

        size = tensor.get("size", [])
        shape = format_shape(size)
        shape_str = "[" + ", ".join(f"'{s}'" if isinstance(s, str) else str(s) for s in shape) + "]"

        # Print basic info
        print(f'  ValueInfo "{name}": type {dtype_str}, shape {shape_str},')

        # Print additional info from FlatBuffers
        if "layout" in tensor:
            print(f'    Layout: {tensor["layout"]} (original: {tensor.get("layout_ori", "n/a")})')

        if "quant_info" in tensor:
            quant = tensor["quant_info"]
            if quant.get("qmethod") or quant.get("qtype"):
                print(f'    Quantization: {quant["qmethod"]} {quant["qtype"]}')


def print_model_summary(parser) -> None:
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
        # Always parse both FlatBuffers and JSON data
        rknn_parser = RKNNParser(model_path, parse_flatbuffers=True)

        # Print merged model information
        print_merged_model_info(rknn_parser)

        # Print merged IO information
        print_merged_io_info(rknn_parser)
    except ValueError as e:
        print(f"Error: Failed to parse RKNN file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
