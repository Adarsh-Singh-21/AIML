import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import ARTIFACTS, DATA, FEATURES, LABELS, REPORTS, SCHEMA_VERSION, TARGET
from .data import encode_features, prepare_data, read_prepared
from .utils import read_json, sha256, write_json


def candidates():
    return {
        "logistic_C0.1": LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42),
        "logistic_C1": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
        "random_forest_depth6": RandomForestClassifier(n_estimators=160, max_depth=6, min_samples_leaf=5,
                                                       class_weight="balanced", n_jobs=1, random_state=42),
        "random_forest_full": RandomForestClassifier(n_estimators=160, min_samples_leaf=5,
                                                     class_weight="balanced", n_jobs=1, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7,
                                                                  learning_rate=0.05, l2_regularization=2,
                                                                  class_weight="balanced", random_state=42),
    }


def metrics(truth, predicted, scores):
    accuracy = float(accuracy_score(truth, predicted))
    n, z = len(truth), 1.95996398454
    denominator = 1 + z*z/n
    centre = (accuracy + z*z/(2*n))/denominator
    radius = z*math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n))/denominator
    return {"records": n, "accuracy": accuracy,
            "accuracy_wilson_95": [centre-radius, centre+radius],
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, scores)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
            "per_class": classification_report(truth, predicted, labels=[0, 1], target_names=list(LABELS),
                                                output_dict=True, zero_division=0)}


def train_run(run_name="numerical_v1", data_dir=DATA, reports=REPORTS, artifacts=ARTIFACTS):
    if Path(run_name).name != run_name or run_name in {".", "..", ""}:
        raise ValueError("Run name must be a single directory name.")
    artifacts, reports = Path(artifacts), Path(reports)
    run = artifacts / run_name
    if run.exists():
        raise ValueError("This run already exists. Use a new --run-name; existing model and results are preserved.")
    prepare_data(data_dir, reports)
    frame, summary = read_prepared(data_dir, reports)
    training = frame.loc[frame.split == "train"]
    x, y = training.loc[:, list(FEATURES)], training[TARGET]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc"}
    comparison, fitted = [], {}
    started = time.perf_counter()
    for name, classifier in candidates().items():
        # Each fold fits its own scaler; no test or validation preprocessing leakage.
        pipeline = Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
        scores = cross_validate(pipeline, x, y, cv=folds, scoring=scoring, n_jobs=1, error_score="raise")
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}"] = float(scores[f"test_{metric}"].mean())
            row[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
        comparison.append(row)
        fitted[name] = pipeline
        print(f"{name}: CV macro F1={row['cv_macro_f1']:.4f}, ROC AUC={row['cv_roc_auc']:.4f}", flush=True)
    comparison.sort(key=lambda row: (-row["cv_macro_f1"], row["model"]))
    winner = comparison[0]["model"]
    model = fitted[winner].fit(x, y)
    run.mkdir(parents=True)
    joblib.dump(model, run / "model.joblib", compress=3)
    reloaded = joblib.load(run / "model.joblib")
    if not np.allclose(model.predict_proba(x.iloc[:5]), reloaded.predict_proba(x.iloc[:5])):
        raise ValueError("Saved-model reloading changed numerical predictions.")
    test_metrics = None
    if not (reports / "test_evaluation.json").exists():
        # The candidate is now frozen. Evaluate the reserved partition once.
        testing = frame.loc[frame.split == "test"]
        test_x, test_y = testing.loc[:, list(FEATURES)], testing[TARGET]
        probabilities = model.predict_proba(test_x)[:, 1]
        prediction = (probabilities >= 0.5).astype(int)
        test_metrics = metrics(test_y, prediction, probabilities)
        dummy = DummyClassifier(strategy="most_frequent").fit(x, y)
        baseline = metrics(test_y, dummy.predict(test_x), dummy.predict_proba(test_x)[:, 1])
        test_metrics.update({"model": winner, "run": run_name, "split_id": summary["split_id"],
                             "threshold": 0.5, "baseline": baseline, "evaluation": "single fixed held-out row-level test"})
        write_json(reports / "test_evaluation.json", test_metrics)
        prediction_table = testing.loc[:, ["row_id", TARGET]].copy()
        prediction_table["predicted_label"] = [LABELS[value] for value in prediction]
        prediction_table["yes_model_score"] = probabilities
        prediction_table.to_csv(run / "test_predictions.csv", index=False)
    metadata = {"schema_version": SCHEMA_VERSION, "model": winner, "run": run_name, "features": list(FEATURES),
                "labels": list(LABELS), "age_range": summary["age_range"], "split_id": summary["split_id"],
                "csv_sha256": summary["csv_sha256"], "model_sha256": sha256(run / "model.joblib"),
                "sklearn_version": sklearn.__version__, "training_records": len(training), "cv_folds": 5,
                "selection_metric": "five-fold training-only macro F1; name breaks ties", "comparison": comparison,
                "training_seconds": time.perf_counter()-started, "created_utc": datetime.now(timezone.utc).isoformat(),
                "device": "CPU", "test_metrics": test_metrics,
                "classifier_parameters": model.named_steps["classifier"].get_params(),
                "quality_note": "Exploratory numerical dataset-label classification; clinical performance is unverified.",
                "evaluation_note": "Held-out test evaluated after CV selection." if test_metrics else
                                   "CV only: this dataset's test split was previously viewed; no fresh independent test score."}
    if comparison[0]["cv_balanced_accuracy"] < 0.60 or (test_metrics and test_metrics["balanced_accuracy"] < 0.60):
        metadata["quality_note"] = "Performance is near chance. This dataset/model has not demonstrated reliable predictive skill."
    write_json(run / "metadata.json", metadata)
    pd.DataFrame(comparison).to_csv(run / "cv_comparison.csv", index=False)
    write_json(artifacts / "active.json", {"run": run_name, "model_sha256": metadata["model_sha256"],
                                           "split_id": metadata["split_id"], "schema_version": SCHEMA_VERSION})
    return metadata


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
