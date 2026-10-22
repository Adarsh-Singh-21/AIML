import tempfile
from pathlib import Path


def pytest_configure(config):
    folder = Path(__file__).resolve().parents[1] / ".cache/test_tmp"
    folder.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(folder)
