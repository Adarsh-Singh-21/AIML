"""Saved-model loading and inference; offline ML code lives in notebooks/train.ipynb."""
from pathlib import Path

import joblib
import numpy as np
import sklearn

from . import ARTIFACTS, DATA, FEATURES, LABELS, REPORTS, SCHEMA_VERSION
from .data import encode_features
from .utils import read_json, sha256


def train_run(run_name="numerical_v1", data_dir=DATA, reports=REPORTS, artifacts=ARTIFACTS):
    """Run the training notebook when the user requests the existing CLI command."""
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("Run name must be a single directory name.")
    if (Path(artifacts) / run_name).exists():
        raise ValueError("This run already exists. Use a new run name.")
    from .notebook_runner import run_workflow
    return run_workflow(run_name, data_dir, reports, artifacts)


def resolve_run(artifacts=ARTIFACTS):
    artifacts = Path(artifacts).resolve()
    pointer = read_json(artifacts / "active.json")
    run = (artifacts / pointer["run"]).resolve()
    if Path(pointer["run"]).is_absolute() or not run.is_relative_to(artifacts):
        raise ValueError("Invalid active numerical model path.")
    metadata = read_json(run / "metadata.json")
    if any(pointer[key] != metadata[key] for key in ("model_sha256", "split_id", "schema_version")):
        raise ValueError("Active model pointer does not match its metadata.")
    return run, metadata


def load_model(run):
    run = Path(run)
    metadata = read_json(run / "metadata.json")
    if metadata["schema_version"] != SCHEMA_VERSION or metadata["features"] != list(FEATURES) or metadata["labels"] != list(LABELS):
        raise ValueError("Model feature schema or target labels are incompatible.")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("scikit-learn version differs from the trained model; use requirements.txt or retrain.")
    if sha256(run / "model.joblib") != metadata["model_sha256"]:
        raise ValueError("Numerical model integrity check failed.")
    # Only load trusted, hash-checked local artifacts; uploaded models are never accepted.
    model = joblib.load(run / "model.joblib")
    if list(model.feature_names_in_) != list(FEATURES) or list(model.classes_) != [0, 1]:
        raise ValueError("Stored estimator feature order or class order is incompatible.")
    return model, metadata


def predict_records(frame, model, metadata, encoding="numeric"):
    inputs = encode_features(frame, encoding=encoding, age_range=tuple(metadata["age_range"]))
    scores = model.predict_proba(inputs)[:, 1]
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Model produced invalid scores.")
    result = inputs.reset_index(drop=True).copy()
    result["predicted_dataset_label"] = np.where(scores >= 0.5, "YES", "NO")
    result["no_model_score"] = 1-scores
    result["yes_model_score"] = scores
    result["model_run"] = metadata["run"]
    result["model_sha256"] = metadata["model_sha256"]
    return result
