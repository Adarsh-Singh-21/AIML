"""Local numerical-data application. No image decoding or CNN inference."""
import hashlib
import io
import json

import pandas as pd
import streamlit as st

from lung_tabular import ARTIFACTS, BINARY_FEATURES, DATASET_URL, FEATURES, REPORTS, ROOT
from lung_tabular.model import load_model, predict_records, resolve_run
from lung_tabular.utils import read_json

st.set_page_config(page_title="Numerical Lung Cancer Project", page_icon="🫁", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:2rem}h1{font-size:2.2rem}</style>", unsafe_allow_html=True)
st.title("Lung Cancer · Numerical Data")
st.caption("Age and 14 coded questionnaire fields · trained on the Akash Nath Kaggle dataset · CPU inference")
st.info("Academic prototype. Predicts the dataset's YES/NO labels. Model scores are not an individual's probability of cancer.")


@st.cache_resource
def saved_model(path, digest):
    return load_model(path)


try:
    run, metadata = resolve_run(ARTIFACTS)
    model, metadata = saved_model(str(run), metadata["model_sha256"])
except FileNotFoundError:
    st.warning("No numerical model is installed yet.")
    st.code("python -m lung_tabular download-data\npython -m lung_tabular prepare-data\npython -m lung_tabular train")
    st.stop()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Numerical model could not be loaded: {error}")
    st.stop()

test = metadata.get("test_metrics")
if test and test["balanced_accuracy"] < 0.60:
    st.warning(f"Limited predictive performance: test accuracy {test['accuracy']:.1%}, compared with "
               f"{test['baseline']['accuracy']:.1%} by always choosing the majority label. "
               "The model has not demonstrated reliable predictive skill. Use this as a numerical ML demonstration.")
else:
    st.warning(metadata["quality_note"])

record_tab, batch_tab, results_tab, project_tab = st.tabs(["Single record", "CSV batch", "Training & results", "Project details"])

with record_tab:
    st.subheader("Enter a numerical record")
    st.caption("Use the dataset's codes: binary input 0 represents source code 1; input 1 represents source code 2. "
               "Gender uses 0=F and 1=M. Binary codes are not symptom severity levels.")
    examples_file = ROOT / "samples/numerical/examples.json"
    tools = st.columns(3)
    if examples_file.exists():
        examples = read_json(examples_file)["examples"]
        for index, example in enumerate(examples[:2]):
            if tools[index].button(f"Load example {index + 1}", key=f"example_{index}"):
                for feature in FEATURES:
                    st.session_state[f"field_{feature}"] = int(example["inputs"][feature])
                st.session_state.pop("record_result", None)
    if tools[2].button("Clear inputs", key="clear_inputs"):
        for feature in FEATURES:
            st.session_state[f"field_{feature}"] = None
        st.session_state.pop("record_result", None)
    demographics = st.columns(2)
    values = {}
    values["AGE"] = demographics[0].number_input("Age (years)", min_value=metadata["age_range"][0],
                                                max_value=metadata["age_range"][1], value=None, step=1,
                                                key="field_AGE", placeholder="Enter age")
    values["GENDER"] = demographics[1].selectbox("Gender code", [0, 1], index=None, key="field_GENDER",
                                               format_func=lambda value: "0 — F" if value == 0 else "1 — M",
                                               placeholder="Choose a code")
    st.markdown("#### Questionnaire features")
    columns = st.columns(3)
    for index, feature in enumerate(BINARY_FEATURES):
        values[feature] = columns[index % 3].selectbox(feature.replace("_", " ").capitalize(), [0, 1], index=None,
                         key=f"field_{feature}", placeholder="Choose a code",
                         format_func=lambda value: f"{value} (source code {value + 1})")
    token = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest() + metadata["model_sha256"]
    if st.session_state.get("record_token") != token:
        st.session_state.pop("record_result", None)
    complete = all(value is not None for value in values.values())
    if st.button("Predict dataset label", key="predict_record", type="primary", disabled=not complete) and complete:
        try:
            result = predict_records(pd.DataFrame([values]), model, metadata).iloc[0].to_dict()
            st.session_state["record_result"] = result
            st.session_state["record_token"] = token
        except (ValueError, TypeError) as error:
            st.session_state.pop("record_result", None)
            st.error(str(error))
    if not complete:
        st.caption("Complete all 15 inputs, or load an example, to enable prediction.")
    result = st.session_state.get("record_result")
    if result:
        st.metric("Predicted dataset label", result["predicted_dataset_label"])
        scores = pd.DataFrame({"Dataset label": ["NO", "YES"], "Model score": [result["no_model_score"], result["yes_model_score"]]})
        st.bar_chart(scores.set_index("Dataset label"), color="#0F766E", height=220)
        st.dataframe(scores.style.format({"Model score": "{:.1%}"}), hide_index=True, width="stretch")
        st.caption("This is an experimental classification of the input record, not a diagnosis or a calibrated personal risk estimate.")
        download = {"prediction": result, "interpretation": "Dataset-label prediction only", "quality_note": metadata["quality_note"]}
        st.download_button("Download result (JSON)", json.dumps(download, indent=2), file_name="numerical_prediction.json",
                           mime="application/json", key="record_download")

with batch_tab:
    st.subheader("Predict from numerical CSV records")
    st.caption("The CSV must have exactly the 15 input columns. Do not include LUNG_CANCER, names, or patient identifiers. "
               "Maximum 10 MB and 10,000 records. Uploaded records are processed in memory and are not saved.")
    template = pd.DataFrame(columns=list(FEATURES)).to_csv(index=False)
    st.download_button("Download empty numeric CSV template", template, "numerical_input_template.csv", mime="text/csv", key="template")
    example_csv = ROOT / "samples/numerical/inputs.csv"
    if example_csv.exists():
        st.download_button("Download example numeric CSV", example_csv.read_bytes(), "numerical_example.csv", mime="text/csv", key="example_csv")
    encoding = st.selectbox("CSV encoding", ["numeric", "survey"], key="batch_encoding",
                           format_func=lambda value: "Numeric: binary 0/1; gender F=0, M=1" if value == "numeric"
                           else "Original survey: binary 1/2; gender F/M")
    upload = st.file_uploader("Upload numerical CSV", type=["csv"], key="numerical_csv")
    if upload is None:
        st.session_state.pop("batch_result", None)
    else:
        try:
            payload = upload.getvalue()
            if len(payload) > 10 * 1024 * 1024:
                raise ValueError("CSV is larger than the 10 MB limit.")
            batch_token = hashlib.sha256(payload).hexdigest() + encoding + metadata["model_sha256"]
            if st.session_state.get("batch_token") != batch_token:
                st.session_state.pop("batch_result", None)
            frame = pd.read_csv(io.BytesIO(payload))
            st.dataframe(frame.head(5), hide_index=True, width="stretch")
            if st.button("Run CSV prediction", key="predict_batch", type="primary"):
                st.session_state["batch_result"] = predict_records(frame, model, metadata, encoding)
                st.session_state["batch_token"] = batch_token
            output = st.session_state.get("batch_result")
            if output is not None:
                st.success(f"Processed {len(output):,} numerical records.")
                st.dataframe(output, hide_index=True, width="stretch")
                st.download_button("Download predictions (CSV)", output.to_csv(index=False), "numerical_predictions.csv",
                                   mime="text/csv", key="batch_download")
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
            st.session_state.pop("batch_result", None)
            st.error(f"CSV could not be predicted: {error}")

with results_tab:
    st.subheader("Numerical model training and evaluation")
    first, second, third, fourth = st.columns(4)
    first.metric("Training records", metadata["training_records"])
    second.metric("Cross-validation folds", metadata["cv_folds"])
    third.metric("Training time (CPU)", f"{metadata['training_seconds']:.1f} seconds")
    fourth.metric("Input features", len(FEATURES))
    st.write(f"Selected model: **{metadata['model']}**. Selection used training-only cross-validation macro F1.")
    st.dataframe(pd.DataFrame(metadata["comparison"]), hide_index=True, width="stretch")
    if test:
        st.markdown("#### Held-out test results")
        st.caption(f"{test['records']} fixed test records, evaluated after model selection. These are row-level results; "
                   "independent patient identity and clinical ground truth are unverified.")
        metrics_columns = st.columns(4)
        for column, (label, value) in zip(metrics_columns, [("Test accuracy", f"{test['accuracy']:.1%}"),
            ("Balanced accuracy", f"{test['balanced_accuracy']:.1%}"), ("Macro F1", f"{test['macro_f1']:.3f}"),
            ("ROC AUC", f"{test['roc_auc']:.3f}")]):
            column.metric(label, value)
        st.write(f"Majority-label baseline accuracy: **{test['baseline']['accuracy']:.1%}**. "
                 f"Accuracy 95% Wilson interval: **{test['accuracy_wilson_95'][0]:.1%}–{test['accuracy_wilson_95'][1]:.1%}**.")
        st.dataframe(pd.DataFrame(test["confusion_matrix"], index=["Actual NO", "Actual YES"],
                                 columns=["Predicted NO", "Predicted YES"]), width="stretch")
        st.dataframe(pd.DataFrame({name: test["per_class"][name] for name in ("NO", "YES")}).T, width="stretch")
    else:
        st.info(metadata["evaluation_note"])

with project_tab:
    st.subheader("How the numerical project works")
    st.write("Numerical record → validated feature codes → saved preprocessing → trained tabular classifier → YES/NO dataset-label scores")
    st.markdown(f"Dataset: [Akash Nath — Lung Cancer Dataset]({DATASET_URL}), version 2. "
                "The application uses the actual CSV fields, not the publisher description's references to vital signs or treatment data.")
    summary_path = REPORTS / "dataset_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        st.write(f"{summary['raw_rows']:,} source records; {summary['duplicate_rows_removed']} duplicates and "
                 f"{summary['conflicting_rows_removed']} conflicting-label records excluded; {summary['usable_rows']:,} usable records.")
        st.dataframe(pd.DataFrame(summary["split_counts"]), width="stretch")
        st.caption(f"Source license as recorded by Kaggle: {summary['source']['license']}.")
    st.markdown("#### Input dictionary")
    st.dataframe(pd.DataFrame({"Feature": list(FEATURES), "Numeric representation": [
        f"Whole years; supported range {metadata['age_range'][0]}–{metadata['age_range'][1]}", "F=0, M=1",
        *["Source code 1→0, source code 2→1"] * len(BINARY_FEATURES)]}), hide_index=True, width="stretch")
    st.caption("The actual CSV does not provide a complete clinical dictionary for each binary code. The app preserves the source coding without inventing severity levels.")
    st.markdown("#### Reproduce training")
    st.code("powershell -ExecutionPolicy Bypass -File .\\train.ps1 -RunName numerical_repeat")
    st.caption("Once the test partition has been viewed, later runs report training CV only. A new unbiased evaluation requires a new independent dataset.")
    for file, label in [(ROOT / "reports/FINAL_REPORT.md", "Download numerical project report"),
                        (ROOT / "docs/MODEL_CARD.md", "Download numerical model card")]:
        if file.exists():
            st.download_button(label, file.read_bytes(), file_name=file.name, key=f"doc_{file.stem}")
    st.caption(f"Model SHA-256: {metadata['model_sha256']}. No images or CT classifier are used in this application.")
