import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import BINARY_FEATURES, DATA, DATASET_URL, FEATURES, REPORTS, SCHEMA_VERSION, TARGET
from .utils import read_json, sha256, write_json


def download_data(destination=DATA):
    destination = Path(destination)
    if (destination / "lung_cancer.csv").exists():
        source = read_json(destination / "source.json")
        if sha256(destination / "lung_cancer.csv") != source["csv_sha256"]:
            raise ValueError("The downloaded numerical CSV has changed.")
        return source
    metadata_url = "https://www.kaggle.com/api/v1/datasets/view/akashnath29/lung-cancer-dataset"
    url = "https://www.kaggle.com/api/v1/datasets/download/akashnath29/lung-cancer-dataset?datasetVersionNumber=2"

    def fetch(address):
        request = urllib.request.Request(address, headers={"User-Agent": "LungNumericalStudentProject/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    metadata = json.loads(fetch(metadata_url))
    payload = fetch(url)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(files) != 1:
            raise ValueError("Expected exactly one dataset CSV in the Kaggle archive.")
        csv = archive.read(files[0])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "lung_cancer.csv").write_bytes(csv)
    source = {"source_page": DATASET_URL, "metadata_url": metadata_url, "download_url": url, "version": 2,
              "license": metadata.get("licenseName"), "publisher": "Akash Nath (Kaggle: akashnath29)",
              "publisher_description": metadata.get("description"), "archive_member": files[0],
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "csv_sha256": hashlib.sha256(csv).hexdigest(), "archive_sha256": hashlib.sha256(payload).hexdigest(),
              "limitation": "Clinical ground truth, patient identity, and collection procedures are unverified."}
    write_json(destination / "source.json", source)
    return source


def normalize_columns(frame):
    frame = frame.copy()
    frame.columns = ["_".join(str(column).strip().upper().split()) for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    return frame


def encode_features(frame, encoding="numeric", age_range=(18, 100)):
    """Strict explicit encodings: numeric=0/1; survey=1/2 and F/M. Never infer codes."""
    frame = normalize_columns(frame)
    missing, extra = set(FEATURES) - set(frame), set(frame) - set(FEATURES)
    if missing or extra:
        raise ValueError(f"Column mismatch. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}. "
                         "Use the 15-column template without a target or patient name.")
    if not 1 <= len(frame) <= 10000:
        raise ValueError("Provide between 1 and 10,000 records.")
    if frame.isna().any().any():
        raise ValueError("All 15 inputs are required. Missing values are not silently filled.")
    result = pd.DataFrame(index=frame.index)
    result["AGE"] = pd.to_numeric(frame["AGE"], errors="coerce")
    if encoding == "survey":
        result["GENDER"] = frame["GENDER"].astype(str).str.strip().str.upper().map({"F": 0, "M": 1})
        for column in BINARY_FEATURES:
            value = pd.to_numeric(frame[column], errors="coerce")
            if not value.isin([1, 2]).all():
                raise ValueError(f"{column}: survey encoding requires 1=No or 2=Yes.")
            result[column] = value - 1
    elif encoding == "numeric":
        for column in ("GENDER", *BINARY_FEATURES):
            result[column] = pd.to_numeric(frame[column], errors="coerce")
    else:
        raise ValueError("Encoding must be numeric or survey.")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Inputs must contain finite numeric values; gender must use the selected encoding.")
    if not (result["AGE"].between(*age_range) & (result["AGE"] % 1 == 0)).all():
        raise ValueError(f"Age must be a whole number between {age_range[0]} and {age_range[1]} for this dataset.")
    for column in ("GENDER", *BINARY_FEATURES):
        if not result[column].isin([0, 1]).all():
            raise ValueError(f"{column}: numeric encoding requires 0 or 1. Use survey encoding for original 1/2 codes.")
    return result.loc[:, list(FEATURES)].astype("int64")


def prepare_data(data_dir=DATA, reports=REPORTS):
    """Run the notebook's data-preparation section for the existing CLI command."""
    from .notebook_runner import run_workflow
    return run_workflow(data_dir=data_dir, reports=reports, prepare_only=True)


def read_prepared(data_dir=DATA, reports=REPORTS):
    data_dir, reports = Path(data_dir), Path(reports)
    summary = read_json(reports / "dataset_summary.json")
    if sha256(data_dir / "lung_cancer.csv") != summary["csv_sha256"]:
        raise ValueError("The numerical dataset changed after preparation.")
    frame = pd.read_csv(reports / "prepared_records.csv")
    actual = hashlib.sha256(json.dumps(frame.to_dict(orient="records"), sort_keys=True).encode()).hexdigest()
    if actual != summary["split_id"]:
        raise ValueError("Prepared records or fixed splits changed after preparation.")
    return frame, summary
