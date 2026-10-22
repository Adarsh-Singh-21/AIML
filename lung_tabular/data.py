import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    data_dir, reports = Path(data_dir), Path(reports)
    csv = data_dir / "lung_cancer.csv"
    source = read_json(data_dir / "source.json")
    if sha256(csv) != source["csv_sha256"]:
        raise ValueError("Raw CSV no longer matches the recorded dataset hash.")
    if (reports / "dataset_summary.json").exists():
        _, summary = read_prepared(data_dir, reports)
        return summary
    raw = normalize_columns(pd.read_csv(csv))
    if TARGET not in raw:
        raise ValueError("Dataset is missing LUNG_CANCER.")
    target = raw[TARGET].astype(str).str.strip().str.upper()
    if not target.isin(["NO", "YES"]).all():
        raise ValueError("Targets must be YES or NO; unknown labels are not remapped.")
    encoded = encode_features(raw.drop(columns=TARGET), "survey")
    encoded[TARGET] = target.map({"NO": 0, "YES": 1})
    encoded["row_id"] = [hashlib.sha256(json.dumps([int(v) for v in values]).encode()).hexdigest()
                         for values in encoded.loc[:, list(FEATURES)].itertuples(index=False, name=None)]
    conflict_ids = set(encoded.groupby("row_id")[TARGET].nunique().loc[lambda values: values > 1].index)
    conflicting = encoded[encoded["row_id"].isin(conflict_ids)]
    remaining = encoded[~encoded["row_id"].isin(conflict_ids)]
    duplicates = remaining[remaining.duplicated("row_id", keep="first")]
    clean = remaining.drop_duplicates("row_id").sort_values("row_id").reset_index(drop=True)
    train_ids, test_ids = train_test_split(clean.row_id, test_size=0.20, stratify=clean[TARGET], random_state=42)
    clean["split"] = np.where(clean.row_id.isin(set(train_ids)), "train", "test")
    split_id = hashlib.sha256(json.dumps(clean.to_dict(orient="records"), sort_keys=True).encode()).hexdigest()
    summary = {"schema_version": SCHEMA_VERSION, "split_id": split_id, "csv_sha256": source["csv_sha256"],
               "source": source, "raw_rows": len(raw), "usable_rows": len(clean),
               "duplicate_rows_removed": len(duplicates), "conflicting_rows_removed": len(conflicting),
               "conflicting_feature_groups": len(conflict_ids), "features": list(FEATURES),
               "age_range": [int(encoded.AGE.min()), int(encoded.AGE.max())], "seed": 42,
               "split_counts": {split: {label: int(((clean.split == split) & (clean[TARGET] == value)).sum())
                                        for value, label in enumerate(("NO", "YES"))} for split in ("train", "test")},
               "limitation": "No verified patient IDs or external validation; source label provenance is unverified."}
    reports.mkdir(parents=True, exist_ok=True)
    clean.to_csv(reports / "prepared_records.csv", index=False)
    exclusions = pd.concat([conflicting.assign(reason="conflicting label for identical features"),
                            duplicates.assign(reason="duplicate feature record")])
    exclusions.loc[:, ["row_id", "reason"]].to_csv(reports / "exclusions.csv", index=False)
    write_json(reports / "dataset_summary.json", summary)
    return summary


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
