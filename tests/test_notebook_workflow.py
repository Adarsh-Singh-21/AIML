"""Notebook migration contracts: offline code must not run during inference."""
from lung_tabular import ROOT


def test_runtime_prediction_does_not_require_notebook(monkeypatch):
    import pandas as pd
    import lung_tabular.notebook_runner as runner
    from lung_tabular.model import load_model, predict_records, resolve_run

    def offline_code_must_not_run(*args, **kwargs):
        raise AssertionError("Application inference must not load or execute the training notebook")

    monkeypatch.setattr(runner, "run_workflow", offline_code_must_not_run)
    run, _ = resolve_run()
    model, metadata = load_model(run)
    inputs = pd.read_csv(ROOT / "samples/numerical/inputs.csv")
    results = predict_records(inputs, model, metadata)
    assert len(results) == len(inputs)
    assert results["predicted_dataset_label"].isin(["NO", "YES"]).all()
