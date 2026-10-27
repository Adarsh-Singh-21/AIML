"""Compatibility bridge to the offline workflow defined in notebooks/train.ipynb.

Only tagged imports and function definitions are loaded. Demonstration cells,
training calls and plotting cells are never run by importing this module.
The Streamlit prediction path does not use this bridge.
"""
import ast
import json
from functools import lru_cache
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/train.ipynb"


@lru_cache(maxsize=2)
def _definitions(source):
    notebook = json.loads(source)
    namespace = {"__name__": __name__}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code" or "workflow-definition" not in cell.get("metadata", {}).get("tags", []):
            continue
        code = cell["source"]
        code = "".join(code) if isinstance(code, list) else code
        filename = f"{NOTEBOOK}:cell-{index + 1}"
        tree = ast.parse(code, filename=filename)
        if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body):
            raise ValueError("Notebook workflow-definition cells must contain only imports and function definitions.")
        exec(compile(tree, filename, "exec"), namespace)
    expected = {"prepare_data", "candidates", "metrics", "train_run", "generate_diagnostics"}
    if not all(callable(namespace.get(name)) for name in expected):
        raise ValueError("The training notebook is missing required workflow definitions.")
    return namespace


def workflow_function(name):
    """Return an offline function without executing notebook example cells."""
    return _definitions(NOTEBOOK.read_text(encoding="utf-8"))[name]
