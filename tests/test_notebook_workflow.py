"""Notebook migration contracts: offline code must not run during inference."""
import json

from lung_tabular import ROOT
from lung_tabular.notebook_workflow import NOTEBOOK, _definitions


def test_bridge_ignores_untagged_execution_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"].append({"cell_type": "code", "metadata": {},
                              "source": "raise RuntimeError('Notebook examples must not run in the CLI bridge')"})
    functions = _definitions(json.dumps(notebook))
    result = functions["metrics"]([0, 1], [0, 1], [0.1, 0.9])
    assert result["accuracy"] == 1
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]


def test_runtime_prediction_does_not_require_notebook(monkeypatch):
    import pandas as pd
    import lung_tabular.notebook_workflow as bridge
    from lung_tabular.model import load_model, predict_records, resolve_run

    def offline_code_must_not_run(*args, **kwargs):
        raise AssertionError("Application inference must not load or execute the training notebook")

    monkeypatch.setattr(bridge, "workflow_function", offline_code_must_not_run)
    run, _ = resolve_run()
    model, metadata = load_model(run)
    inputs = pd.read_csv(ROOT / "samples/numerical/inputs.csv")
    results = predict_records(inputs, model, metadata)
    assert len(results) == len(inputs)
    assert results["predicted_dataset_label"].isin(["NO", "YES"]).all()
