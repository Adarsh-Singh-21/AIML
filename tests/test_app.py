import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

import lung_tabular
from lung_tabular import FEATURES, SCHEMA_VERSION
from lung_tabular.utils import sha256, write_json

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app_fixture(tmp_path, monkeypatch):
    """A small synthetic estimator exercises software flow, not medical model quality."""
    artifacts = tmp_path / "artifacts/numerical"
    run = artifacts / "fixture"
    run.mkdir(parents=True)
    generator = np.random.default_rng(42)
    x = pd.DataFrame(generator.integers(0, 2, (40, len(FEATURES))), columns=list(FEATURES))
    x["AGE"] = generator.integers(30, 81, 40)
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression())]).fit(x, [0, 1] * 20)
    joblib.dump(model, run / "model.joblib")
    metadata = {"schema_version": SCHEMA_VERSION, "model": "fixture_logistic", "run": "fixture",
                "features": list(FEATURES), "labels": ["NO", "YES"], "age_range": [30, 80],
                "split_id": "fixture", "model_sha256": sha256(run / "model.joblib"),
                "sklearn_version": sklearn.__version__, "training_records": 40, "cv_folds": 5,
                "comparison": [{"model": "fixture_logistic", "cv_macro_f1": 0.5}],
                "training_seconds": 0.1, "test_metrics": None, "quality_note": "Software test fixture only",
                "evaluation_note": "CV-only fixture"}
    write_json(run / "metadata.json", metadata)
    write_json(artifacts / "active.json", {key: metadata[key] for key in ("run", "model_sha256", "split_id", "schema_version")})
    write_json(tmp_path / "samples/numerical/examples.json", {"examples": [{"inputs": x.iloc[0].to_dict()}, {"inputs": x.iloc[1].to_dict()}]})
    monkeypatch.setattr(lung_tabular, "ROOT", tmp_path)
    monkeypatch.setattr(lung_tabular, "ARTIFACTS", artifacts)
    monkeypatch.setattr(lung_tabular, "REPORTS", tmp_path / "reports/numerical")
    return x


def test_missing_numerical_model_is_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(lung_tabular, "ARTIFACTS", tmp_path / "missing")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("No numerical model" in warning.value for warning in app.warning)


def test_record_completeness_predictions_and_state_reset(app_fixture):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.button(key="predict_record").disabled
    assert len(app.number_input) == 1
    assert not any("image" in uploader.label.lower() for uploader in app.get("file_uploader"))
    app.button(key="example_0").click().run()
    assert not app.exception and not app.button(key="predict_record").disabled
    app.button(key="predict_record").click().run()
    assert not app.exception and app.session_state["record_result"]["predicted_dataset_label"] in ("NO", "YES")
    age = app.number_input(key="field_AGE").value
    app.number_input(key="field_AGE").set_value(30 if age != 30 else 31).run()
    assert "record_result" not in app.session_state
    app.button(key="clear_inputs").click().run()
    assert app.button(key="predict_record").disabled and "record_result" not in app.session_state


def test_batch_prediction_and_invalid_csv_clear_results(app_fixture, monkeypatch):
    import streamlit as st
    payload = {"content": app_fixture.head(2).to_csv(index=False).encode()}

    class Upload:
        def getvalue(self):
            return payload["content"]

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload())
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key="predict_batch").click().run()
    assert not app.exception
    assert len(app.session_state["batch_result"]) == 2
    payload["content"] = b"AGE,UNEXPECTED\n55,1\n"
    app.run()
    assert "batch_result" not in app.session_state
    app.button(key="predict_batch").click().run()
    assert not app.exception and any("Column mismatch" in error.value for error in app.error)
