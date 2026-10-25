# Lung Cancer Classification Using Numerical Data

A complete academic tabular ML project using the exact [Akash Nath dataset](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset) requested by the user. Inputs are age and 14 coded questionnaire fields. The model predicts the source CSV's **NO/YES** labels. No images, CNN, or CT recognition are used.

**Measured limitation:** selected model **hist_gradient_boosting**, test accuracy **50.92%**, macro F1 **0.508**, ROC AUC **0.531**. Majority-label baseline accuracy is **50.58%**. This evaluation shows little predictive skill. The interface makes this explicit; these scores do not justify medical use.

**Developed by:** Adarsh Raj Choudhary, Adarsh Singh, Adith K P.

**Submitted to:** Prof. Vaishali Ingale.

The Evaluation page uses the saved test results for the selected model and baseline, with candidate cross-validation shown separately. Its training-only permutation importance is a model-reliance diagnostic, not clinical relevance or causality. Regenerate it with `python -m lung_tabular.diagnostics`.

## Run on this laptop

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Open http://localhost:8501. The sidebar groups Home, Data, Preprocessing & EDA, Training & Tuning, Evaluation, Model Insights, Single-record prediction, and CSV batch prediction. Enter all 15 inputs or load an example on the prediction page, then select Predict dataset label. The CSV batch page provides a template. Changing inputs clears previous predictions. Uploaded records are processed in memory and are not saved.

## Deploy on AWS

Use the [AWS EC2 deployment template and guide](deploy/aws/README.md). The CloudFormation template installs Python 3.12, verifies the saved model, and runs Streamlit behind Nginx as a service. For zero-charge hosting, first verify the AWS Free account plan; resources consume its credits and access ends when the plan or credits expire. The template creates no IAM role. The initial deployment provides an HTTP demonstration URL; a custom domain and HTTPS are separate configuration steps.

## Set up a fresh machine

Python 3.12 is supported. This is a **CPU-only** scikit-learn application; CUDA and PyTorch are not required.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m lung_tabular doctor
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

requirements.txt pins direct dependencies; requirements.lock.txt records the numerical project's installed Windows dependency closure. The existing environment may still contain historical CT dependencies, but the active project does not import them.

## Data and encodings

The downloaded Kaggle version is **2**: **3,000 rows**, 15 inputs and LUNG_CANCER. Two duplicate rows and four rows from two conflicting-label feature groups are excluded, leaving **2,994 records**. Fixed seed 42 assigns **2,395 training** and **599 held-out test** records. Identical feature profiles cannot cross the partitions. These are records, not verified unique patients.

For the app and numeric CSV: AGE is an integer in the supported source range **30–80**; GENDER is F=0, M=1; each other field maps source code 1→0 and source code 2→1. Source binary codes are not treated as severity. Do not assume that the CSV supplies a clinically validated symptom codebook. The original survey encoding can be explicitly selected for batch input. Encodings are never guessed.

| Input | Numerical representation |
|---|---|
| AGE | Whole years, 30–80 |
| GENDER | F=0; M=1 |
| SMOKING | Source 1→0; source 2→1 |
| YELLOW_FINGERS | Source 1→0; source 2→1 |
| ANXIETY | Source 1→0; source 2→1 |
| PEER_PRESSURE | Source 1→0; source 2→1 |
| CHRONIC_DISEASE | Source 1→0; source 2→1 |
| FATIGUE | Source 1→0; source 2→1 |
| ALLERGY | Source 1→0; source 2→1 |
| WHEEZING | Source 1→0; source 2→1 |
| ALCOHOL_CONSUMING | Source 1→0; source 2→1 |
| COUGHING | Source 1→0; source 2→1 |
| SHORTNESS_OF_BREATH | Source 1→0; source 2→1 |
| SWALLOWING_DIFFICULTY | Source 1→0; source 2→1 |
| CHEST_PAIN | Source 1→0; source 2→1 |

Prediction CSVs require **exactly these 15 columns**, without the target, personal identifiers, missing entries, or extra fields. Files are limited to 10 MB and 10,000 records. Invalid values, unsupported ages, images renamed as CSV, and ambiguous 1/2 data in numeric mode are rejected.

## Training and evaluation

```powershell
.\.venv\Scripts\python.exe -m lung_tabular download-data
.\.venv\Scripts\python.exe -m lung_tabular prepare-data
powershell -ExecutionPolicy Bypass -File .\train.ps1 -RunName my_numerical_run
```

The five predefined candidates are two regularized logistic regressions, two random forests, and histogram gradient boosting. Five-fold stratified CV uses only training records. StandardScaler is inside each Pipeline and is fitted separately in each fold. The highest CV macro F1 wins; the final estimator fits the training partition. The initial selected run was evaluated once on the reserved test partition with a fixed 0.5 threshold. Test scores were not used for candidate selection. Training and initial evaluation took **7.02 seconds on CPU**.

Subsequent runs preserve the viewed test evaluation and report CV only; they do not claim a new independent test score. Existing run folders and split manifests are never silently overwritten. New training activates that numerical run. Model candidates are defined in lung_tabular/model.py. To reuse the included trained model, just run the app.

```powershell
.\.venv\Scripts\python.exe -m lung_tabular predict --csv samples/numerical/inputs.csv --output predictions.csv
```

Use `--encoding survey` for original F/M and 1/2 input codes. `python -m lung_ai` forwards to the numerical CLI for compatibility. Saved local models are hash-checked before loading; never replace them with untrusted pickle/joblib files. The app does not accept model uploads.

## Project files

- app.py and ui_pages/: grouped sidebar navigation, Home, model-building pages, Evaluation, and numerical prediction pages.
- lung_tabular/: strict input schema, dataset audit, training, prediction, reporting, and packaging.
- data/numerical/akashnath29/: requested dataset and provenance.
- artifacts/numerical/: active pointer, saved Pipeline, metadata, CV scores, and frozen test predictions.
- reports/numerical/: fixed records/splits, exclusions, evaluation, plots, and verification.
- notebooks/train.ipynb: executable numerical workflow; included model reused unless training is explicitly requested.
- docs/: dataset card, model card, and presentation/demo guide.
- legacy_ct/: preserved historical CT project, excluded from this project's app, tests, and new package.

## Verification and delivery

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest_numerical
.\.venv\Scripts\python.exe -m lung_tabular.reporting
.\.venv\Scripts\python.exe -m lung_tabular.package
```

The final [report](reports/FINAL_REPORT.md) and [model card](docs/MODEL_CARD.md) explain the measured limitations. Software checks are recorded in reports/numerical/VERIFICATION.json. Submission is submission/Lung_Numerical_Project.zip, including numerical source, trained model, dataset, notebook, and documents; historical CT data and the installed environment are excluded.

## Attribution and limitations

Source: Akash Nath (Kaggle: akashnath29), [Lung Cancer Dataset](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset), version 2. Kaggle records the license as **Database: Open Database, Contents: © Original Authors**. Preserve the source attribution and applicable dataset terms. Dataset SHA-256: `417052af60ea9da985478d93d6be49749d25c8d0168a14bbd9f94515ab80c6ce`. Full source metadata is saved next to the CSV. The previously considered 309-row dataset is not used for this model.

Clinical ground truth, individual patient identity, recruitment, and collection procedures are unverified. Source descriptions mention fields absent from the actual CSV, so only observed columns are used. Binary codes are preserved without assuming clinical severity or inventing a codebook. This is a retrospective dataset-label classifier, not a future-risk model, screening recommendation, or diagnosis. Scores are not calibrated individual cancer probabilities. No independent external or patient-level validation is established.
