"""Run the notebook for optional offline commands; the app does not use this file."""
import sys
from pathlib import Path

from . import ARTIFACTS, DATA, REPORTS, ROOT
from .utils import read_json


def run_workflow(run_name=None, data_dir=DATA, reports=REPORTS, artifacts=ARTIFACTS,
                 prepare_only=False, diagnostics=False, output=None):
    # These notebook tools are needed only for preprocessing/training commands.
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(ROOT / "notebooks/train.ipynb", as_version=4)
    reports, artifacts = Path(reports), Path(artifacts)

    # Apply the command's settings without changing the original notebook file.
    for cell in notebook.cells:
        if cell.id == "settings":
            cell.source = (
                f"TRAIN_NEW_MODEL = {run_name is not None!r}\n"
                f"RUN_NAME = {run_name!r}\n"
                f"REGENERATE_DIAGNOSTICS = {diagnostics!r}\n"
                f"DATA = Path({str(Path(data_dir).resolve())!r})\n"
                f"REPORTS = Path({str(reports.resolve())!r})\n"
                f"ARTIFACTS = Path({str(artifacts.resolve())!r})\n"
            )

    # Data preparation stops before the model-training section.
    if prepare_only:
        last = next(index for index, cell in enumerate(notebook.cells) if cell.id == "prepared-data")
        notebook.cells = notebook.cells[:last + 1]

    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=kernel, timeout=180,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    except CellExecutionError as error:
        raise ValueError(f"Notebook execution failed: {error}") from error

    output = Path(output) if output else ROOT / ".cache/last_notebook_run.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)

    if prepare_only:
        return read_json(reports / "dataset_summary.json")
    if diagnostics:
        return read_json(reports / "feature_importance.json")
    if run_name is None:
        run_name = read_json(artifacts / "active.json")["run"]
    return read_json(artifacts / run_name / "metadata.json")
