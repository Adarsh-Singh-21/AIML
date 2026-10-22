"""Build a portable numerical-only project archive and verify every file hash."""
import json
import os
import zipfile
from pathlib import PurePosixPath

from . import ROOT
from .utils import sha256, write_json

DIRECTORIES = ("lung_tabular", "lung_ai", "ui_pages", "tests", "docs", "notebooks", "samples/numerical",
               "data/numerical/akashnath29", "artifacts/numerical", "reports/numerical", ".streamlit")
FILES = ("app.py", "README.md", "PROJECT_PLAN.md", "requirements.txt", "requirements.lock.txt",
         "setup.ps1", "run.ps1", "train.ps1", "pytest.ini", "reports/FINAL_REPORT.md")


def build_package():
    files = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        for base, directories, names in os.walk(ROOT / name):
            directories[:] = [item for item in directories if item not in {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}]
            files.extend(ROOT / base / item for item in names if not item.endswith((".pyc", ".tmp")))
    if not (ROOT / "reports/numerical/VERIFICATION.json").exists():
        raise ValueError("Complete numerical project verification before packaging.")
    destination = ROOT / "submission/Lung_Numerical_Project.zip"
    destination.parent.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():
                raise ValueError(f"Missing or unsafe file: {path}")
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"Lung_Numerical_Project/{relative}")
            manifest[relative] = sha256(path)
        archive.writestr("Lung_Numerical_Project/PACKAGE_MANIFEST.json", json.dumps({
            "project": "numerical", "source_dataset": "akashnath29/lung-cancer-dataset version 2",
            "files": manifest, "excluded": ["legacy_ct", ".venv", ".cache", "historical CT ZIPs"]}, indent=2))
    with zipfile.ZipFile(destination) as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive path.")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed.")
        import hashlib
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(f"Lung_Numerical_Project/{name}")).hexdigest() != digest:
                raise ValueError(f"Archive content hash mismatch: {name}")
    result = {"archive": str(destination), "size_mib": round(destination.stat().st_size/1024**2, 2),
              "sha256": sha256(destination), "files_verified": len(manifest), "integrity": "passed"}
    write_json(destination.with_name(destination.stem + "_info.json"), result)
    return result


if __name__ == "__main__":
    print(json.dumps(build_package(), indent=2))
