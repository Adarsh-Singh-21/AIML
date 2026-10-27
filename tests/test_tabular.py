import json

import numpy as np
import pandas as pd
import pytest

from lung_tabular import BINARY_FEATURES, FEATURES, TARGET
from lung_tabular.data import encode_features, prepare_data, read_prepared
from lung_tabular.model import load_model, predict_records, resolve_run, train_run
from lung_tabular.utils import read_json, sha256, write_json


def input_record():
    return {"AGE": 55, "GENDER": 0, **{name: index % 2 for index, name in enumerate(BINARY_FEATURES)}}


@pytest.fixture
def raw_dataset(tmp_path):
    """Synthetic rows are software fixtures, never part of medical model training."""
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index in range(80):
        row = {"AGE": 30 + index % 51, "GENDER": "F" if index % 2 == 0 else "M"}
        row.update({name: 1 + ((index >> bit) & 1) for bit, name in enumerate(BINARY_FEATURES)})
        row[TARGET] = "YES" if index % 2 else "NO"
        records.append(row)
    records.append(records[0].copy())
    conflicting = records[1].copy()
    conflicting[TARGET] = "NO"
    records.append(conflicting)
    pd.DataFrame(records).to_csv(data / "lung_cancer.csv", index=False)
    write_json(data / "source.json", {"csv_sha256": sha256(data / "lung_cancer.csv"), "source_page": "software fixture"})
    return data, tmp_path / "reports", tmp_path / "artifacts"


def test_explicit_encodings_round_trip_and_column_order():
    numeric = pd.DataFrame([input_record()])
    survey = numeric.copy()
    survey["GENDER"] = "F"
    survey.loc[:, list(BINARY_FEATURES)] += 1
    assert encode_features(numeric).equals(encode_features(survey, "survey"))
    assert list(encode_features(numeric.loc[:, list(reversed(FEATURES))])) == list(FEATURES)
    with pytest.raises(ValueError, match="0 or 1"):
        encode_features(survey.assign(GENDER=0))


@pytest.mark.parametrize("changes", [{"AGE": 101}, {"AGE": 42.5}, {"SMOKING": 2}, {"GENDER": "X"},
                                      {"ANXIETY": np.nan}, {"AGE": np.inf}])
def test_invalid_numerical_values_are_rejected(changes):
    with pytest.raises(ValueError):
        encode_features(pd.DataFrame([{**input_record(), **changes}]))


def test_schema_prevents_target_leakage_and_missing_fields():
    record = pd.DataFrame([input_record()])
    for bad in (record.assign(LUNG_CANCER="YES"), record.drop(columns="AGE"), record.assign(PATIENT_NAME="name")):
        with pytest.raises(ValueError, match="Column mismatch"):
            encode_features(bad)
    with pytest.raises(ValueError, match="between 1 and"):
        encode_features(record.iloc[:0])


def test_duplicate_conflict_audit_and_fixed_split_integrity(raw_dataset):
    data, reports, _ = raw_dataset
    summary = prepare_data(data, reports)
    assert summary["raw_rows"] == 82 and summary["usable_rows"] == 79
    assert summary["duplicate_rows_removed"] == 1 and summary["conflicting_rows_removed"] == 2
    frame, _ = read_prepared(data, reports)
    assert frame.row_id.is_unique
    assert set(frame.loc[frame.split == "train", "row_id"]).isdisjoint(frame.loc[frame.split == "test", "row_id"])
    repeat = reports.parent / "repeat"
    assert prepare_data(data, repeat)["split_id"] == summary["split_id"]
    original = (reports / "prepared_records.csv").read_bytes()
    changed = pd.read_csv(reports / "prepared_records.csv")
    changed.loc[0, "AGE"] += 1
    changed.to_csv(reports / "prepared_records.csv", index=False)
    with pytest.raises(ValueError, match="fixed splits changed"):
        read_prepared(data, reports)
    (reports / "prepared_records.csv").write_bytes(original)
    with (data / "lung_cancer.csv").open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ValueError, match="dataset changed"):
        read_prepared(data, reports)


def test_training_reload_prediction_and_frozen_test(raw_dataset):
    data, reports, artifacts = raw_dataset
    metadata = train_run("first", data, reports, artifacts)
    frame, _ = read_prepared(data, reports)
    run, selected = resolve_run(artifacts)
    model, loaded = load_model(run)
    assert selected["model_sha256"] == metadata["model_sha256"]
    assert model.named_steps["scale"].n_samples_seen_ == int((frame.split == "train").sum())
    record = frame.loc[frame.split == "train", list(FEATURES)].head(2)
    predicted = predict_records(record, model, loaded)
    assert np.allclose(predicted.yes_model_score, model.predict_proba(record)[:, 1])
    assert np.allclose(predicted.yes_model_score + predicted.no_model_score, 1)
    with pytest.raises(ValueError, match="already exists"):
        train_run("first", data, reports, artifacts)
    frozen = (reports / "test_evaluation.json").read_bytes()
    next_metadata = train_run("second", data, reports, artifacts)
    assert next_metadata["test_metrics"] is None
    assert (reports / "test_evaluation.json").read_bytes() == frozen
    pointer = read_json(artifacts / "active.json")
    pointer["run"] = "../outside"
    write_json(artifacts / "active.json", pointer)
    with pytest.raises(ValueError, match="Invalid active"):
        resolve_run(artifacts)
    with (run / "model.joblib").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        load_model(run)
