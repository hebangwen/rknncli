"""RKNN CLI - A command line tool for parsing and displaying RKNN model information."""

def _get_version():
    """Get version from package metadata or pyproject.toml."""
    # 方法1: 从包元数据获取（已安装时）
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version(__name__)
    except (ImportError, ModuleNotFoundError, importlib_metadata.PackageNotFoundError):
        pass

    # 方法2: 从 pyproject.toml 获取（开发时）
    try:
        import os

        # 找到 pyproject.toml 文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pyproject_path = os.path.join(current_dir, "..", "pyproject.toml")
        pyproject_path = os.path.abspath(pyproject_path)

        if os.path.exists(pyproject_path):
            # 手动解析 pyproject.toml 中的版本
            with open(pyproject_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("version = "):
                        # 提取版本号
                        version = line.split("=")[1].strip()
                        # 移除引号
                        return version.strip('"\'')
    except Exception as e:
        print(f"从 pyproject.toml 读取版本失败: {e}")


__version__ = _get_version()
