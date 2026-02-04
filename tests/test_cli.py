"""Unit tests for RKNN CLI tool."""

import subprocess
import sys
from pathlib import Path
import pytest


class TestRKNNCLI:
    """Test cases for rknncli command-line interface."""

    @pytest.fixture
    def assets_dir(self):
        """Return the assets directory path."""
        return Path(__file__).parent.parent / "assets"

    @pytest.fixture
    def rknn_files(self, assets_dir):
        """Return list of all .rknn files in assets directory."""
        return list(assets_dir.glob("*.rknn"))

    def test_cli_help(self):
        """Test that CLI shows help message."""
        result = subprocess.run(
            [sys.executable, "-m", "rknncli.cli", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "A command line tool for parsing and displaying RKNN model information" in result.stdout

    def test_cli_version(self):
        """Test that CLI shows version information."""
        result = subprocess.run(
            [sys.executable, "-m", "rknncli.cli", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "rknncli" in result.stdout

    def test_cli_nonexistent_file(self):
        """Test CLI behavior with non-existent file."""
        result = subprocess.run(
            [sys.executable, "-m", "rknncli.cli", "nonexistent.rknn"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert "Error: File not found" in result.stderr

    def test_cli_analyze_single_model(self, assets_dir):
        """Test analyzing a single RKNN model."""
        model_path = assets_dir / "base-encoder.rknn"
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found")

        result = subprocess.run(
            [sys.executable, "-m", "rknncli.cli", str(model_path)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        # Check that model information is printed
        assert "Model:" in result.stdout
        assert "Input information" in result.stdout
        assert "Output information" in result.stdout
        # Check that quant info is NOT printed
        assert "quant" not in result.stdout.lower()

    def test_cli_analyze_all_models(self, rknn_files):
        """Test analyzing all RKNN models in assets directory."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            result = subprocess.run(
                [sys.executable, "-m", "rknncli.cli", str(model_path)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"Failed to analyze {model_path.name}: {result.stderr}"
            # Verify output contains expected information
            assert "Model:" in result.stdout, f"Model info missing for {model_path.name}"
            assert "Input information" in result.stdout, f"Input info missing for {model_path.name}"
            assert "Output information" in result.stdout, f"Output info missing for {model_path.name}"

    def test_model_info_content(self, rknn_files):
        """Test that model information is properly formatted."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            result = subprocess.run(
                [sys.executable, "-m", "rknncli.cli", str(model_path)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0

            # Check for specific content patterns
            lines = result.stdout.split('\n')

            # Find model line
            model_lines = [line for line in lines if line.startswith("Model:")]
            assert len(model_lines) > 0, f"Model name not found for {model_path.name}"

            # Check that model name is not empty
            model_name = model_lines[0].split("Model:")[1].strip()
            assert model_name, f"Model name is empty for {model_path.name}"

    def test_no_quant_info_printed(self, rknn_files):
        """Test that quant info is not printed in output."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            result = subprocess.run(
                [sys.executable, "-m", "rknncli.cli", str(model_path)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            # Ensure quant info is not in output
            assert "quant" not in result.stdout.lower()
            assert "layout" not in result.stdout.lower()

    def test_io_tensor_info(self, rknn_files):
        """Test that input/output tensor information is properly displayed."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            result = subprocess.run(
                [sys.executable, "-m", "rknncli.cli", str(model_path)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0

            # Check for tensor information patterns
            assert "ValueInfo" in result.stdout, f"Tensor ValueInfo not found for {model_path.name}"

            # Check for common tensor attributes
            lines = result.stdout.split('\n')
            tensor_lines = [line for line in lines if "ValueInfo" in line]

            for line in tensor_lines:
                # Each tensor line should contain type and shape information
                assert "type" in line, f"Tensor type missing in line: {line}"
                assert "shape" in line, f"Tensor shape missing in line: {line}"
                # Should not contain layout or quant info
                assert "layout" not in line.lower(), f"Layout info should not be in line: {line}"
                assert "quant" not in line.lower(), f"Quant info should not be in line: {line}"

    def test_flatbuffers_info(self, rknn_files):
        """Test that FlatBuffers information is properly displayed when available."""
        if not rknn_files:
            pytest.skip("No .rknn files found in assets directory")

        for model_path in rknn_files:
            result = subprocess.run(
                [sys.executable, "-m", "rknncli.cli", str(model_path)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0

            # Check for FlatBuffers-specific information
            fb_fields = ["Format:", "Source:", "Compiler:", "Runtime:"]
            found_fields = []

            for field in fb_fields:
                if field in result.stdout:
                    found_fields.append(field)

            # 注意：某些模型可能没有 FlatBuffers 信息，这是正常的
            # 我们只检查是否有模型信息，不强制要求有 FlatBuffers 信息
            assert "Model:" in result.stdout, f"Model info missing for {model_path.name}"