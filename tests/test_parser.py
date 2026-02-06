"""Unit tests for RKNN parser module."""

import pytest
from pathlib import Path

from rknncli.graph import Graph as ComputeGraph
from rknncli.parser import RKNNParser


class TestRKNNParser:
    """Test cases for RKNNParser class."""

    @pytest.fixture
    def assets_dir(self):
        """Return the assets directory path."""
        return Path(__file__).parent.parent / "assets"

    @pytest.fixture
    def rknn_files(self, assets_dir):
        """Return list of all .rknn files in assets directory."""
        return list(assets_dir.glob("*.rknn"))

    def test_parser_initialization(self, rknn_files):
        """Test parser initialization with valid RKNN files."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path, parse_flatbuffers=True)
            assert parser.file_path == model_path
            assert parser.header is not None
            assert parser.model_info is not None

    def test_parser_invalid_file(self):
        """Test parser behavior with invalid file."""
        with pytest.raises(ValueError, match="Invalid magic number"):
            RKNNParser(__file__)  # Use this Python file as invalid RKNN file

    def test_parser_nonexistent_file(self):
        """Test parser behavior with non-existent file."""
        with pytest.raises(FileNotFoundError):
            RKNNParser("nonexistent.rknn")

    def test_get_model_name(self, rknn_files):
        """Test getting model name from RKNN files."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path)
            model_name = parser.get_model_name()
            assert isinstance(model_name, str)
            # Model name should not be empty
            assert len(model_name) > 0

    def test_get_version(self, rknn_files):
        """Test getting version from RKNN files."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path)
            version = parser.get_version()
            assert isinstance(version, str)

    def test_get_target_platform(self, rknn_files):
        """Test getting target platform from RKNN files."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path)
            platforms = parser.get_target_platform()
            assert isinstance(platforms, list)

    def test_get_flatbuffers_info(self, rknn_files):
        """Test getting FlatBuffers information."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path, parse_flatbuffers=True)
            fb_info = parser.get_flatbuffers_info()

            if fb_info:  # FlatBuffers info might not be available for all models
                assert isinstance(fb_info, dict)
                # Check for expected keys
                expected_keys = ["format", "source", "compiler", "runtime", "num_graphs"]
                for key in expected_keys:
                    if key in fb_info:
                        assert isinstance(fb_info[key], (str, int))

    def test_get_generator_info(self, rknn_files):
        """Test getting generator information (placeholder function)."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path, parse_flatbuffers=True)
            generator_info = parser.get_generator_info()

            # 可能返回 None（如果没有 generator 信息）或 dict（如果有）
            assert generator_info is None or isinstance(generator_info, dict)

    def test_get_input_info(self, rknn_files):
        """Test getting input information."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path)
            inputs = parser.get_input_info()

            assert isinstance(inputs, list)

            # Check input tensors
            for tensor in inputs:
                assert isinstance(tensor, dict)
                assert "tensor_id" in tensor
                assert "dtype" in tensor
                assert "size" in tensor

    def test_get_output_info(self, rknn_files):
        """Test getting output information."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path)
            outputs = parser.get_output_info()

            assert isinstance(outputs, list)

            # Check output tensors
            for tensor in outputs:
                assert isinstance(tensor, dict)
                assert "tensor_id" in tensor
                assert "dtype" in tensor
                assert "size" in tensor

    def test_tensor_info_consistency(self, rknn_files):
        """Test that tensor information is consistent."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path, parse_flatbuffers=True)
            inputs, outputs = parser.get_merged_io_info()

            # Check that all tensors have required fields
            all_tensors = inputs + outputs
            for tensor in all_tensors:
                # Verify tensor_id is non-negative
                assert tensor["tensor_id"] >= 0

                # Verify size is a list
                assert isinstance(tensor["size"], list)

                # Verify dtype is present
                assert tensor["dtype"] is not None

    def test_parser_without_flatbuffers(self, rknn_files):
        """Test parser without FlatBuffers parsing."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path, parse_flatbuffers=False)

            # Basic info should still be available
            model_name = parser.get_model_name()
            assert isinstance(model_name, str)

            # FlatBuffers info should be None or empty
            fb_info = parser.get_flatbuffers_info()
            assert fb_info is None or fb_info == {}

    @pytest.mark.parametrize("parse_fb", [True, False])
    def test_parser_flatbuffers_option(self, rknn_files, parse_fb):
        """Test parser with different FlatBuffers parsing options."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            parser = RKNNParser(model_path, parse_flatbuffers=parse_fb)

            # Should not raise any exceptions
            model_name = parser.get_model_name()
            version = parser.get_version()
            platforms = parser.get_target_platform()

            assert isinstance(model_name, str)
            assert isinstance(version, str)
            assert isinstance(platforms, list)

    def test_render_graphviz_svg(self, rknn_files, tmp_path, monkeypatch):
        """Test that Graphviz SVG rendering writes output."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        model_path = rknn_files[0]
        parser = RKNNParser(model_path, parse_flatbuffers=True)
        if not parser.fb_model or parser.fb_model.GraphsLength() == 0:
            pytest.skip("FlatBuffers graph not available for this model")

        def fake_render_svg(self, output_path):
            output = Path(output_path).with_suffix(".svg")
            output.write_text("<svg></svg>", encoding="utf-8")
            return str(output)

        monkeypatch.setattr(ComputeGraph, "render_svg", fake_render_svg)

        output_path = tmp_path / "graph.svg"
        rendered = parser.render_graphviz(output_path)

        assert rendered.exists()
        assert rendered.suffix == ".svg"

    def test_build_compute_graph(self, rknn_files):
        """Test that parser builds compute graph instance."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        model_path = rknn_files[0]
        parser = RKNNParser(model_path, parse_flatbuffers=True)
        if not parser.fb_model or parser.fb_model.GraphsLength() == 0:
            pytest.skip("FlatBuffers graph not available for this model")

        graph = parser.build_graph()
        assert isinstance(graph, ComputeGraph)

    def test_build_vpmn_graph(self, assets_dir):
        """Test building a graph from VPMN-based RKNN model."""
        model_path = assets_dir / "yolov5s_relu_rv1109_rv1126_out_opt.rknn"
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        parser = RKNNParser(model_path, parse_flatbuffers=True)
        graph = parser.build_graph()
        assert isinstance(graph, ComputeGraph)
        assert len(graph.nodes) > 0

    def test_build_json_graph(self, assets_dir):
        """Test building a graph from extracted JSON model."""
        model_path = assets_dir / "test.json"
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        parser = RKNNParser(model_path, parse_flatbuffers=False)
        graph = parser.build_graph()
        assert isinstance(graph, ComputeGraph)
        assert len(graph.nodes) > 0
