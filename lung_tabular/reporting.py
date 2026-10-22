"""Generate documentation and scientific plots from the saved numerical experiment."""
import os
from pathlib import Path

import pandas as pd

from . import ARTIFACTS, BINARY_FEATURES, DATASET_URL, FEATURES, REPORTS, ROOT
from .model import load_model, resolve_run
from .utils import read_json, write_json


def generate():
    # This report documents the single evaluated run, even if a later CV-only run
    # becomes active. Never attach the old test score to a different model.
    evaluation = read_json(REPORTS / "test_evaluation.json")
    run = ARTIFACTS / evaluation["run"]
    metadata = read_json(run / "metadata.json")
    model, _ = load_model(run)
    metadata["classifier_parameters"] = model.named_steps["classifier"].get_params()
    write_json(run / "metadata.json", metadata)
    summary = read_json(REPORTS / "dataset_summary.json")
    test = metadata["test_metrics"]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    source = summary["source"]
    dictionary = "\n".join(f"| {name} | {'Whole years, 30–80' if name == 'AGE' else 'F=0; M=1' if name == 'GENDER' else 'Source 1→0; source 2→1'} |" for name in FEATURES)
    limitations = ("Clinical ground truth, individual patient identity, recruitment, and collection procedures are unverified. "
                   "Source descriptions mention fields absent from the actual CSV, so only observed columns are used. "
                   "Binary codes are preserved without assuming clinical severity or inventing a codebook. "
                   "This is a retrospective dataset-label classifier, not a future-risk model, screening recommendation, or diagnosis. "
                   "Scores are not calibrated individual cancer probabilities. No independent external or patient-level validation is established.")
    readme = f"""# Lung Cancer Classification Using Numerical Data

A complete academic tabular ML project using the exact [Akash Nath dataset]({DATASET_URL}) requested by the user. Inputs are age and 14 coded questionnaire fields. The model predicts the source CSV's **NO/YES** labels. No images, CNN, or CT recognition are used.

**Measured limitation:** selected model **{metadata['model']}**, test accuracy **{test['accuracy']:.2%}**, macro F1 **{test['macro_f1']:.3f}**, ROC AUC **{test['roc_auc']:.3f}**. Majority-label baseline accuracy is **{test['baseline']['accuracy']:.2%}**. This evaluation shows little predictive skill. The interface makes this explicit; these scores do not justify medical use.

## Run on this laptop

```powershell
powershell -ExecutionPolicy Bypass -File .\\run.ps1
```

Open http://localhost:8501. In Single record, enter all 15 inputs or load an example, then select Predict dataset label. The CSV batch tab accepts numerical records and provides a template. Changing inputs clears previous predictions. Uploaded records are processed in memory and are not saved.

## Set up a fresh machine

Python 3.12 is supported. This is a **CPU-only** scikit-learn application; CUDA and PyTorch are not required.

```powershell
powershell -ExecutionPolicy Bypass -File .\\setup.ps1
.\\.venv\\Scripts\\python.exe -m lung_tabular doctor
powershell -ExecutionPolicy Bypass -File .\\run.ps1
```

requirements.txt pins direct dependencies; requirements.lock.txt records the numerical project's installed Windows dependency closure. The existing environment may still contain historical CT dependencies, but the active project does not import them.

## Data and encodings

The downloaded Kaggle version is **2**: **{summary['raw_rows']:,} rows**, 15 inputs and LUNG_CANCER. Two duplicate rows and four rows from two conflicting-label feature groups are excluded, leaving **{summary['usable_rows']:,} records**. Fixed seed 42 assigns **2,395 training** and **599 held-out test** records. Identical feature profiles cannot cross the partitions. These are records, not verified unique patients.

For the app and numeric CSV: AGE is an integer in the supported source range **30–80**; GENDER is F=0, M=1; each other field maps source code 1→0 and source code 2→1. Source binary codes are not treated as severity. Do not assume that the CSV supplies a clinically validated symptom codebook. The original survey encoding can be explicitly selected for batch input. Encodings are never guessed.

| Input | Numerical representation |
|---|---|
{dictionary}

Prediction CSVs require **exactly these 15 columns**, without the target, personal identifiers, missing entries, or extra fields. Files are limited to 10 MB and 10,000 records. Invalid values, unsupported ages, images renamed as CSV, and ambiguous 1/2 data in numeric mode are rejected.

## Training and evaluation

```powershell
.\\.venv\\Scripts\\python.exe -m lung_tabular download-data
.\\.venv\\Scripts\\python.exe -m lung_tabular prepare-data
powershell -ExecutionPolicy Bypass -File .\\train.ps1 -RunName my_numerical_run
```

The five predefined candidates are two regularized logistic regressions, two random forests, and histogram gradient boosting. Five-fold stratified CV uses only training records. StandardScaler is inside each Pipeline and is fitted separately in each fold. The highest CV macro F1 wins; the final estimator fits the training partition. The initial selected run was evaluated once on the reserved test partition with a fixed 0.5 threshold. Test scores were not used for candidate selection. Training and initial evaluation took **{metadata['training_seconds']:.2f} seconds on CPU**.

Subsequent runs preserve the viewed test evaluation and report CV only; they do not claim a new independent test score. Existing run folders and split manifests are never silently overwritten. New training activates that numerical run. Model candidates are defined in lung_tabular/model.py. To reuse the included trained model, just run the app.

```powershell
.\\.venv\\Scripts\\python.exe -m lung_tabular predict --csv samples/numerical/inputs.csv --output predictions.csv
```

Use `--encoding survey` for original F/M and 1/2 input codes. `python -m lung_ai` forwards to the numerical CLI for compatibility. Saved local models are hash-checked before loading; never replace them with untrusted pickle/joblib files. The app does not accept model uploads.

## Project files

- app.py: numerical single-record and CSV application.
- lung_tabular/: strict input schema, dataset audit, training, prediction, reporting, and packaging.
- data/numerical/akashnath29/: requested dataset and provenance.
- artifacts/numerical/: active pointer, saved Pipeline, metadata, CV scores, and frozen test predictions.
- reports/numerical/: fixed records/splits, exclusions, evaluation, plots, and verification.
- notebooks/train.ipynb: executable numerical workflow; included model reused unless training is explicitly requested.
- docs/: dataset card, model card, and presentation/demo guide.
- legacy_ct/: preserved historical CT project, excluded from this project's app, tests, and new package.

## Verification and delivery

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q --basetemp .cache/pytest_numerical
.\\.venv\\Scripts\\python.exe -m lung_tabular.reporting
.\\.venv\\Scripts\\python.exe -m lung_tabular.package
```

The final [report](reports/FINAL_REPORT.md) and [model card](docs/MODEL_CARD.md) explain the measured limitations. Software checks are recorded in reports/numerical/VERIFICATION.json. Submission is submission/Lung_Numerical_Project.zip, including numerical source, trained model, dataset, notebook, and documents; historical CT data and the installed environment are excluded.

## Attribution and limitations

Source: {source['publisher']}, [Lung Cancer Dataset]({DATASET_URL}), version 2. Kaggle records the license as **{source['license']}**. Preserve the source attribution and applicable dataset terms. Dataset SHA-256: `{summary['csv_sha256']}`. Full source metadata is saved next to the CSV. The previously considered 309-row dataset is not used for this model.

{limitations}
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (docs / "DATASET_CARD.md").write_text(f"""# Numerical dataset card

**Source:** [{source['publisher']}]({DATASET_URL}), Kaggle version 2. Filename in archive: dataset.csv.

**Observed CSV:** 3,000 records, 15 input fields, target LUNG_CANCER (NO/YES). Raw target counts: 1,482 NO and 1,518 YES. Actual age range: 30–80. No vital-sign, treatment, scan, or survival columns occur in this CSV, despite broader language in the publisher description.

**Audit:** two duplicate rows removed; four records in two feature-identical but differently labelled groups quarantined. Such conflicts need not be data-entry errors: different people can have the same observed profile. Exclusion is a conservative prototype choice. Patient identifiers are unavailable, so duplicates also cannot be proved to refer to the same person. Usable records: 2,994; training 2,395; test 599. See exclusions.csv and prepared_records.csv.

**Encoding:** AGE retains years. F→0 and M→1 in GENDER. All other source feature codes map 1→0 and 2→1. Numeric values do not imply ordinal symptom severity. A complete validated clinical codebook is not supplied with the CSV; no extra semantics are invented. Target NO→0 and YES→1.

| Field | Representation |
|---|---|
{dictionary}

**Split and integrity:** duplicate/ambiguous profile groups handled before stratified seed-42 80/20 row split. Raw and prepared data hashes are checked before training. Preprocessing is fitted only on training folds. No synthetic records are added to this medical demonstration dataset.

**License label from Kaggle:** {source['license']}. Retain source attribution and applicable publisher terms; this is not the CC0 dataset initially considered.

**SHA-256:** `{summary['csv_sha256']}`.

**Limitations:** {limitations}
""", encoding="utf-8")
    (docs / "MODEL_CARD.md").write_text(f"""# Numerical model card

**Task:** classify the selected Kaggle dataset's NO/YES labels from 15 numerical inputs. No image processing or CT model is used.

**Evaluated run:** {metadata['run']}. **Model:** {metadata['model']}; scikit-learn {metadata['sklearn_version']}. The saved Pipeline includes StandardScaler and the trained estimator. All parameters are learned from the numerical training data; no pretrained predictor is used. Later active CV-only runs do not inherit this test result.

**Protocol:** 2,395 training records; five-fold stratified CV; selection by mean macro F1. CV macro F1 {metadata['comparison'][0]['cv_macro_f1']:.4f}; CV ROC AUC {metadata['comparison'][0]['cv_roc_auc']:.4f}. The selected Pipeline is fitted to training records only. Training time {metadata['training_seconds']:.2f} seconds on CPU. See saved metadata for exact parameters.

**Fixed test:** 599 records; accuracy {test['accuracy']:.2%}; balanced accuracy {test['balanced_accuracy']:.2%}; macro F1 {test['macro_f1']:.4f}; ROC AUC {test['roc_auc']:.4f}. Majority-label baseline accuracy {test['baseline']['accuracy']:.2%}. Accuracy Wilson 95% interval {test['accuracy_wilson_95'][0]:.2%}–{test['accuracy_wilson_95'][1]:.2%}.

**Conclusion:** this evaluation is close to chance and does not demonstrate useful clinical prediction. The application shows this limitation prominently. It does not output low/medium/high medical risk bands or make claims about cancer stage or type.

**Artifact:** artifacts/numerical/{run.name}/model.joblib. SHA-256 `{metadata['model_sha256']}`. Only load trusted local artifacts; uploaded models are never accepted.

**Limitations:** {limitations}
""", encoding="utf-8")
    (docs / "DEMO_GUIDE.md").write_text("""# Numerical project demonstration

1. Start with run.ps1 and open http://localhost:8501.
2. State the task accurately: predicting NO/YES dataset labels from numerical coded inputs.
3. Show the dataset source, actual columns, encoding dictionary, exclusions, and fixed split.
4. Load example 1 in Single record. Explain that it is a training example selected by row hash, not correctness. Predict and download the JSON. Change an input to show the previous result clearing.
5. In CSV batch, download the example, upload it, choose numeric encoding, run prediction, and download the CSV. Show the separate original-source encoding option.
6. Open Training & results. Explain training-only CV selection and held-out evaluation. State the near-chance result and the majority-label baseline; do not present it as a useful cancer detector.
7. Show the notebook and retraining command. Subsequent runs do not repeatedly evaluate the viewed test split.
8. Discuss future work: verify source labels and clinical codebook, obtain an appropriate independent cohort, then evaluate calibration and clinical relevance. More models cannot substitute for informative, verified data.

Suggested slides: objective; requested dataset; encoding and audit; CV pipeline; model comparison; test and baseline; live numerical app; limitations and future work.
""", encoding="utf-8")
    comparison = "\n".join(f"| {row['model']} | {row['cv_macro_f1']:.4f} | {row['cv_roc_auc']:.4f} |" for row in metadata["comparison"])
    report = f"""# Lung Cancer Classification Using Numerical Data

## Abstract

This project replaces CT-image identification with a numerical machine-learning workflow using the user-selected Akash Nath Kaggle dataset, version 2. The CSV has 3,000 records, 15 inputs, and binary NO/YES labels. After duplicate removal and ambiguous-profile quarantine, 2,994 records remain. Five candidate tabular classifiers are compared through five-fold training-only cross-validation. Histogram gradient boosting is selected, saved with its preprocessing, and evaluated on 599 reserved records. Test accuracy is {test['accuracy']:.2%}, compared with {test['baseline']['accuracy']:.2%} for the majority-label baseline. The result is near chance; this is an educational numerical ML implementation with no established clinical predictive utility. A local application supports complete record entry, batch CSV prediction, downloads, results, and documentation.

## 1. Scope and source

The task is supervised classification of the source dataset's LUNG_CANCER labels. Inputs are age, a coded gender field, and 13 coded questionnaire features. There is no image feature extraction, cancer-type classification, stage prediction, or defined future-risk horizon. Source: [Akash Nath dataset]({DATASET_URL}), downloaded version 2. Kaggle license label: {source['license']}. Full provenance and dataset hash are preserved. The publisher description is treated as documentation, not instructions, and the implementation follows the observed CSV rather than references to absent vital signs or treatment data.

## 2. Data audit and preprocessing

Raw labels: 1,482 NO and 1,518 YES. Two duplicate records are removed. Four records from two identical-feature groups containing contradictory labels are quarantined rather than assigning an arbitrary target. Distinct patients could genuinely share these profiles; the dataset supplies no reliable patient identifiers. These decisions leave 2,994 records. Seed 42 stratifies the fixed split: training 1,183 NO + 1,212 YES = 2,395; test 296 NO + 303 YES = 599. Duplicate feature profiles cannot cross partitions.

All inputs become numbers: age in years; GENDER F→0/M→1; other feature codes 1→0/2→1. No missing data or invalid category is silently imputed. Actual column names are normalized, but unexpected fields and target leakage are rejected for prediction. Input ages are restricted to the source range 30–80. StandardScaler is fitted within each cross-validation Pipeline and later on the final training partition only.

## 3. Training and selection

Five predefined candidates use a fixed five-fold stratified training CV. Model selection uses macro F1, with model name as a deterministic tie-breaker. Candidate results:

| Candidate | Mean CV macro F1 | Mean CV ROC AUC |
|---|---:|---:|
{comparison}

The selected estimator is {metadata['model']}, with 120 boosting iterations, at most seven leaves per tree, learning rate 0.05, L2 regularization 2, balanced class weighting, and seed 42. It is fitted only to the 2,395 training records. The full initial comparison, fit, and evaluation take {metadata['training_seconds']:.2f} seconds on CPU. GPU training is unnecessary for this dataset size. Exact estimator parameters and library versions are stored in metadata.

## 4. Held-out evaluation

The selected candidate is frozen before the single reserved-test evaluation. Threshold 0.5 is fixed in advance. Accuracy: {test['accuracy']:.2%}; balanced accuracy: {test['balanced_accuracy']:.2%}; macro F1: {test['macro_f1']:.4f}; ROC AUC: {test['roc_auc']:.4f}. Accuracy Wilson 95% interval: {test['accuracy_wilson_95'][0]:.2%}–{test['accuracy_wilson_95'][1]:.2%}. Majority-label baseline accuracy: {test['baseline']['accuracy']:.2%}. Results are row-level and do not establish patient independence or clinical performance.

Confusion matrix, rows actual NO/YES and columns predicted NO/YES: {test['confusion_matrix']}.

![Numerical test confusion matrix](numerical/confusion_matrix.png)

![ROC from saved test predictions](numerical/roc_curve.png)

The selected model improves accuracy by only {(test['accuracy']-test['baseline']['accuracy'])*100:.2f} percentage points over the majority-label baseline. This is not evidence of a reliable clinical classifier. The interface and model card make the weak measured performance explicit. There is no attempt to improve the displayed test score by tuning against the held-out set. Later training runs preserve this test record and report CV only.

## 5. Application and reproducibility

Single-record entry requires all 15 values and clears predictions when any field changes. CSV prediction accepts an explicit numerical or source-code encoding, rejects target/extra columns and invalid values, processes uploads in memory, and returns numerical features, dataset label, two model scores, run identity, and artifact hash. Scores are not displayed as individual cancer risk. Model loading verifies the hash, schema, feature order, class order, and scikit-learn version. Preprocessing travels with the model Pipeline. The model is bundled for inference without raw training data.

setup.ps1 installs CPU dependencies. run.ps1 starts the local app. train.ps1 trains a new named run without overwriting old results. The notebook reproduces the numerical workflow and reuses saved training by default. Source, model, data provenance, notebook, reports, tests, and setup scripts are packaged together. Software and portability checks are in numerical/VERIFICATION.json.

## 6. Limitations and future work

{limitations}

The source dataset is balanced, but this alone says nothing about real-world prevalence or label validity. The near-chance CV and test results warrant investigating data collection and labels, not making stronger accuracy claims. Future work should verify source records, define a clinically meaningful target and prediction horizon, obtain an independently documented cohort, and assess calibration and external performance. Repeated model tuning does not create information missing from the measured features.

## References

1. Akash Nath. [Lung Cancer Dataset, Kaggle version 2]({DATASET_URL}). Dataset SHA-256 `{summary['csv_sha256']}`.
2. [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), for preprocessing fitted inside CV.
3. [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).
4. [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html), including trusted-artifact and version constraints.
"""
    (ROOT / "reports/FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "PROJECT_PLAN.md").write_text("""# Completed numerical project plan

1. Use the exact user-selected Akash Nath Kaggle dataset, version 2.
2. Audit actual columns, duplicates, ambiguous labels, numeric codes, and provenance.
3. Fix an 80/20 split and compare five tabular candidates using training-only five-fold CV.
4. Freeze selection, evaluate the reserved test once, and preserve measured results honestly.
5. Replace image uploads with complete numerical record and CSV inputs.
6. Include saved preprocessing, schema validation, state clearing, model integrity, and result downloads.
7. Update the report, cards, notebook, setup/retraining scripts, tests, and portable submission.

The current result is near chance. Independent, verified data and a meaningful target are priorities for further work. See reports/FINAL_REPORT.md. The prior CT implementation is preserved in legacy_ct and the earlier image ZIP; it is not part of this numerical application.
""", encoding="utf-8")
    _plots(run, test)
    return {"report": str(ROOT / "reports/FINAL_REPORT.md"), "model": metadata["model"]}


def _plots(run, test):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.imshow(test["confusion_matrix"], cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NO", "YES"], yticklabels=["NO", "YES"],
           xlabel="Predicted dataset label", ylabel="Actual dataset label", title="Numerical model · fixed test records")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(test["confusion_matrix"][i][j]), ha="center", va="center", color="white")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png", dpi=160); plt.close(fig)
    predictions = pd.read_csv(run / "test_predictions.csv")
    fpr, tpr, _ = roc_curve(predictions["LUNG_CANCER"], predictions["yes_model_score"])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.plot(fpr, tpr, label=f"Model ROC AUC {test['roc_auc']:.3f}", color="#0F766E")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Numerical model · held-out ROC", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "roc_curve.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    print(generate())
