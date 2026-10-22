"""Generate training-only model reliance diagnostics without touching test scores."""
from pathlib import Path

from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

from . import DATA, FEATURES, REPORTS, TARGET
from .data import read_prepared
from .model import load_model, resolve_run
from .utils import write_json


def generate_diagnostics(data_dir=DATA, reports=REPORTS):
    run, _ = resolve_run()
    model, metadata = load_model(run)
    frame, summary = read_prepared(data_dir, reports)
    if summary["split_id"] != metadata["split_id"]:
        raise ValueError("Diagnostic records do not match the model's training split.")
    training = frame.loc[frame.split == "train"]
    sample = training.sample(n=min(512, len(training)), random_state=42)
    x, y = sample.loc[:, list(FEATURES)], sample[TARGET]
    result = permutation_importance(model, x, y, scoring="balanced_accuracy", n_repeats=5,
                                    random_state=42, n_jobs=1)
    payload = {"model_sha256": metadata["model_sha256"], "split_id": metadata["split_id"],
               "run": metadata["run"], "partition": "training only", "records": len(sample),
               "repeats": 5, "seed": 42, "scoring": "balanced accuracy",
               "training_subset_score": float(balanced_accuracy_score(y, model.predict(x))),
               "features": [{"feature": name, "importance": float(mean), "std": float(std)}
                            for name, mean, std in zip(FEATURES, result.importances_mean, result.importances_std)],
               "limitation": "In-sample reliance on training records, not external validation, clinical relevance, or causality."}
    write_json(Path(reports) / "feature_importance.json", payload)
    return {"run": metadata["run"], "partition": payload["partition"], "records": len(sample)}


if __name__ == "__main__":
    print(generate_diagnostics())
