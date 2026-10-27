"""Shared Streamlit pages and evidence-based evaluation views."""
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from . import BINARY_FEATURES, DATASET_URL, FEATURES
from .evaluation import cv_table, metric_table, model_name
from .model import predict_records
from .utils import read_json

DEVELOPERS = "Adarsh Raj Choudhary, Adarsh Singh, Adith K P"
PROFESSOR = "Prof. Vaishali Ingale"
EVALUATOR = "Vaishali Ingle"


def context():
    return st.session_state["_project_context"]


def header(title, subtitle):
    st.title(title)
    st.caption(subtitle)


def quality_notice():
    metadata = context()["metadata"]
    test = metadata.get("test_metrics")
    if test and test["balanced_accuracy"] < 0.60:
        st.warning(f"Limited predictive performance: test accuracy {test['accuracy']:.1%}, compared with "
                   f"{test['baseline']['accuracy']:.1%} by always choosing the majority label. "
                   "The model has not demonstrated reliable predictive skill. Use this as a numerical ML demonstration.")
    else:
        st.warning(metadata["quality_note"])


def documents():
    root = context()["root"]
    for path, label in [(root / "reports/FINAL_REPORT.md", "Download project report"),
                        (root / "docs/MODEL_CARD.md", "Download model card"),
                        (root / "docs/DEMO_GUIDE.md", "Download demonstration guide")]:
        if path.exists():
            st.download_button(label, path.read_bytes(), file_name=path.name, key=f"doc_{path.stem}")


def render_home():
    metadata = context()["metadata"]
    header("🫁 Lung Cancer · Numerical Data", "AIML Mini Project · dataset: Akash Nath / Kaggle · numerical questionnaire classification")
    with st.container(border=True):
        st.markdown(f"👥 **Developed by:** {DEVELOPERS}  \n🎓 **Submitted to:** {PROFESSOR}  \n📝 **Evaluator:** {EVALUATOR}")
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.subheader("The problem")
        st.write("A numerical record contains age and coded questionnaire answers. This project demonstrates how a supervised "
                 "learning pipeline can use those inputs to predict the source dataset's NO or YES label.")
        st.write("The workflow answers two practical questions:")
        st.markdown("1. **Is the record complete?** Validate all 15 fields and their numeric codes.\n"
                    "2. **What pattern does the model predict?** Apply saved preprocessing and the trained tabular classifier.")
        st.caption("This dataset does not establish a clinically reliable cancer detector. The model's scores are not personal cancer probabilities.")
    with right:
        st.subheader("Project at a glance")
        st.info("🔢 **Numerical input** — age and 14 coded fields; no image identification.")
        st.success("📋 **Dataset labels** — NO / YES from the selected Kaggle CSV.")
        st.warning("🧪 **Research prototype** — near-chance measured performance; not a medical assessment.")
    st.subheader("How the system works")
    columns = st.columns(4)
    for column, title, text in zip(columns,
        ["1 · Enter data", "2 · Validate", "3 · Apply the model", "4 · Inspect evidence"],
        ["Use a numerical record or CSV.", "Check column names, codes, and age.",
         "Reuse the trained preprocessing pipeline.", "Review label scores, test metrics, and limitations."]):
        with column:
            with st.container(border=True, height="stretch"):
                st.markdown(f"**{title}**")
                st.caption(text)
    action = st.columns(3)
    if action[0].button("Open numerical prediction", type="primary", width="stretch"):
        st.switch_page("ui_pages/predict.py")
    if action[1].button("Explore evaluation", width="stretch"):
        st.switch_page("ui_pages/evaluation.py")
    if action[2].button("View the dataset", width="stretch"):
        st.switch_page("ui_pages/data.py")
    quality_notice()
    with st.expander("Reports and reproducibility"):
        st.caption(f"Active run: {metadata['run']} · {model_name(metadata['model'])} · CPU inference")
        documents()


def input_dictionary():
    metadata = context()["metadata"]
    return pd.DataFrame({"Feature": list(FEATURES), "Numeric representation": [
        f"Whole years; supported range {metadata['age_range'][0]}–{metadata['age_range'][1]}",
        "F=0, M=1", *["Source code 1→0; source code 2→1"] * len(BINARY_FEATURES)]})


def summary():
    path = context()["reports"] / "dataset_summary.json"
    return read_json(path) if path.exists() else None


def training_records():
    path = context()["reports"] / "prepared_records.csv"
    if not path.exists():
        return None
    records = pd.read_csv(path)
    return records.loc[records.split == "train"]


def render_data():
    header("📥 1 · Data", "The exact user-selected numerical dataset and its input dictionary.")
    st.markdown(f"Source: [Akash Nath — Lung Cancer Dataset]({DATASET_URL}), **version 2**.")
    info = summary()
    if info:
        columns = st.columns(4)
        for column, name, value in zip(columns, ["Source records", "Usable records", "Input features", "Target labels"],
                                       [info["raw_rows"], info["usable_rows"], len(FEATURES), "NO / YES"]):
            column.metric(name, value)
        st.dataframe(pd.DataFrame(info["split_counts"]).rename_axis("Dataset label"), width="stretch")
        st.caption(f"Kaggle license label: {info['source']['license']}. Preserve the source attribution and applicable dataset terms.")
        with st.expander("Source integrity and provenance"):
            st.code(f"CSV SHA-256: {info['csv_sha256']}")
            st.write("Patient identity, source clinical labels, and the collection process are not independently verified.")
    st.subheader("Input dictionary")
    st.dataframe(input_dictionary(), hide_index=True, width="stretch")
    st.info("The CSV does not supply a complete clinical codebook. Source binary codes are preserved; they are not treated as severity levels.")
    st.caption("Prediction CSVs must contain these 15 inputs without the LUNG_CANCER target, names, or patient identifiers.")


def render_preprocessing():
    header("🧹 2 · Preprocessing & EDA", "Cleaning, fixed splits, encoding, and a view of training records only.")
    info = summary()
    if info:
        columns = st.columns(3)
        for column, label, value in zip(columns, ["Duplicates removed", "Ambiguous records excluded", "Usable records"],
                  [info["duplicate_rows_removed"], info["conflicting_rows_removed"], info["usable_rows"]]):
            column.metric(label, value)
    st.markdown("1. Normalize column names and validate the original codes.\n"
                "2. Deduplicate exact feature profiles and quarantine profiles with conflicting labels.\n"
                "3. Create a fixed, stratified 80/20 row split using seed 42.\n"
                "4. Fit StandardScaler inside each training CV fold, then with the final training partition.")
    st.caption("Neither the test rows nor their labels are used to fit preprocessing. Missing inputs are not silently filled.")
    records = training_records()
    if records is not None:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Training label distribution")
            counts = records.LUNG_CANCER.map({0: "NO", 1: "YES"}).value_counts().sort_index()
            st.bar_chart(counts.rename("Records"), color="#327CCB", x_label="Dataset label", height=290)
        with right:
            st.subheader("Training age distribution")
            ages = pd.cut(records.AGE, bins=range(30, 86, 5), right=False).value_counts(sort=False)
            ages.index = [f"{int(group.left)}–{int(group.right)-1}" for group in ages.index]
            st.bar_chart(ages.rename("Records"), color="#24A581", x_label="Age in years", height=290)
    else:
        st.info("Prepared training records are not included in this installation. Prediction remains available from the saved model.")


def render_training():
    metadata = context()["metadata"]
    header("🧠 3 · Training & Tuning", "Five predefined candidates compared through training-only cross-validation.")
    columns = st.columns(4)
    for column, name, value in zip(columns, ["Training records", "CV folds", "Training time (CPU)", "Candidates"],
              [metadata["training_records"], metadata["cv_folds"], f"{metadata['training_seconds']:.1f} s", len(metadata["comparison"])]):
        column.metric(name, value)
    st.markdown(f"Selected by CV macro F1: **{model_name(metadata['model'])}**")
    render_cv()
    with st.expander("Selected estimator parameters"):
        st.json(metadata.get("classifier_parameters", {}))
    st.subheader("Reproduce training")
    st.code("powershell -ExecutionPolicy Bypass -File .\\train.ps1 -RunName numerical_repeat")
    st.caption("Existing runs are preserved. After the original test partition has been viewed, later runs report CV only. "
               "A new unbiased evaluation requires an independent dataset.")


def render_cv():
    metadata = context()["metadata"]
    frame = cv_table(metadata["comparison"])
    st.dataframe(frame.style.format("{:.4f}"), width="stretch")
    st.caption("Training-only stratified five-fold CV. These scores are not held-out test results. "
               "All candidate preprocessing is fitted inside the folds.")
    if "CV macro F1" in frame:
        st.bar_chart(frame[["CV macro F1"]], horizontal=True, color="#327CCB", x_label="Candidate", y_label="Mean validation macro F1", height=320)


def render_importance():
    ctx = context()
    path = ctx["reports"] / "feature_importance.json"
    if not path.exists():
        st.info("No feature-reliance diagnostics are saved for this model.")
        st.code("python -m lung_tabular.diagnostics")
        return
    evidence = read_json(path)
    if evidence.get("model_sha256") != ctx["metadata"]["model_sha256"] or evidence.get("split_id") != ctx["metadata"]["split_id"]:
        st.info("The saved feature diagnostics belong to a different model. Regenerate them for the active run.")
        return
    st.subheader("Training-only permutation importance")
    st.caption(f"Drop in balanced accuracy after shuffling each feature, using {evidence['records']} training records "
               f"and {evidence['repeats']} shuffles. No test records were used.")
    frame = pd.DataFrame(evidence["features"]).sort_values("importance", ascending=False)
    frame["feature"] = frame.feature.str.replace("_", " ").str.title()
    chart = frame.set_index("feature")[["importance"]].rename(columns={"importance": "Mean score drop"})
    st.bar_chart(chart, horizontal=True, sort=False, color="#347CC8", height=470,
                 x_label="Feature", y_label="Decrease in training balanced accuracy")
    st.dataframe(frame.rename(columns={"feature": "Feature", "importance": "Mean score drop", "std": "Std (5 shuffles)"})
                 .style.format({"Mean score drop": "{:.4f}", "Std (5 shuffles)": "{:.4f}"}), hide_index=True, width="stretch")
    st.warning("This describes reliance on training data, not clinical importance or causality. The model remains near chance on the held-out test.")


def render_class_breakdown(test):
    frame = pd.DataFrame({name: test["per_class"][name] for name in ("NO", "YES")}).T
    frame.index.name = "Actual dataset label"
    st.dataframe(frame.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}), width="stretch")
    st.bar_chart(frame[["precision", "recall", "f1-score"]], stack=False,
                 color=["#327CCB", "#23AA83", "#E4A323"], height=300)
    st.caption("NO and YES are source dataset labels, not verified medical verdicts. No independent external cohort has been evaluated.")


def render_evaluation():
    ctx = context()
    metadata = ctx["metadata"]
    header("📊 4 · Evaluation", "How well does the numerical model perform on records it did not train on?")
    st.markdown(f"Selected by training CV: **{model_name(metadata['model'])}**")
    test = metadata.get("test_metrics")
    if not test:
        st.info(metadata["evaluation_note"])
        render_cv()
        return
    focus = st.session_state.get("evaluation_focus", "YES class")
    table = metric_table(test, focus)
    selected = table.iloc[0]
    columns = st.columns(4)
    for column, name in zip(columns, ["Accuracy", "Precision", "Recall", "F1"]):
        column.metric(name, f"{selected[name]:.3f}")
    st.caption(f"{test['records']} fixed held-out records · precision, recall and F1: {focus.lower()} · "
               "model selection used training CV, not test scores.")
    tabs = st.tabs(["📋 Metrics", "🧮 Confusion matrices", "📈 ROC", "⭐ Feature importance", "🏷️ Class breakdown", "🗂️ Cross-validation"])
    with tabs[0]:
        st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#D1E4F7"), width="stretch")
        st.caption("Held-out comparison is available for the selected model and the majority-label baseline only. "
                   "Other candidates are compared separately in the Cross-validation tab.")
        st.subheader("Model comparison on the held-out test set")
        metric_order = ["Accuracy", "Precision", "Recall", "F1"]
        chart_data = table.reset_index().melt(id_vars="Model", value_vars=metric_order,
                                               var_name="Metric", value_name="Score")
        base = alt.Chart(chart_data).encode(
            y=alt.Y("Metric:N", title=None, sort=metric_order),
            yOffset=alt.YOffset("Model:N", sort=list(table.index)),
            x=alt.X("Score:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1.06]),
                    axis=alt.Axis(values=[0, .2, .4, .6, .8, 1], format=".1f")),
            color=alt.Color("Model:N", scale=alt.Scale(domain=list(table.index), range=["#327CCB", "#EA8533"]),
                            legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=290)),
            tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")])
        bars = base.mark_bar()
        values = base.mark_text(align="left", dx=5).encode(text=alt.Text("Score:Q", format=".3f"), color=alt.value("#364152"))
        chart = (bars + values).properties(height=420).configure_view(stroke=None).configure_axis(gridColor="#EBEDF0")
        st.altair_chart(chart, width="stretch")
        st.download_button("Download evaluation metrics (CSV)", table.to_csv(), "evaluation_metrics.csv", mime="text/csv")
        quality_notice()
    with tabs[1]:
        import matplotlib.pyplot as plt
        st.subheader("Confusion matrices")
        normalize = st.toggle("Show row-normalized percentages", key="normalize_confusion")
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, label, record in zip(axes, table.index, [test, test["baseline"]]):
            import numpy as np
            matrix = np.asarray(record["confusion_matrix"], dtype=float)
            values = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix
            axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, matrix.max()))
            axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NO", "YES"], yticklabels=["NO", "YES"],
                     xlabel="Predicted dataset label", ylabel="Actual dataset label", title=label)
            for row in range(2):
                for col in range(2):
                    text = f"{values[row, col]:.1%}" if normalize else str(int(matrix[row, col]))
                    axis.text(col, row, text, ha="center", va="center",
                              color="white" if values[row, col] > values.max()/2 else "#1F2937")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        st.caption("Derived from the saved confusion counts. Normalization does not run inference or reevaluate the model.")
    with tabs[2]:
        st.subheader("Receiver operating characteristic")
        roc = ctx["reports"] / "roc_curve.png"
        if roc.exists():
            st.image(str(roc), width=650)
        else:
            st.info("The saved ROC plot is not included in this installation.")
        columns = st.columns(2)
        columns[0].metric("Selected model ROC AUC", f"{test['roc_auc']:.3f}")
        columns[1].metric("Chance / baseline ROC AUC", f"{test['baseline']['roc_auc']:.3f}")
        st.caption("The curve is generated from saved test predictions. No new test evaluation is performed by this page.")
    with tabs[3]:
        render_importance()
    with tabs[4]:
        render_class_breakdown(test)
    with tabs[5]:
        st.subheader("Training-only candidate comparison")
        render_cv()


def render_insights():
    header("🔎 5 · Model Insights", "Inspect model reliance and remaining errors without implying clinical relevance.")
    render_importance()
    test = context()["metadata"].get("test_metrics")
    if test:
        st.subheader("Errors on the reserved test partition")
        matrix = test["confusion_matrix"]
        left, right = st.columns(2)
        left.metric("YES records predicted NO", matrix[1][0])
        right.metric("NO records predicted YES", matrix[0][1])
        st.caption("These are source-label classification errors from the frozen evaluation, not independently verified patient outcomes.")


def render_record():
    ctx = context()
    ROOT, REPORTS = ctx['root'], ctx['reports']
    run, model, metadata = ctx['run'], ctx['model'], ctx['metadata']
    header('🩺 Single-record prediction', 'Enter numerical values or load an example record.')
    quality_notice()
    for feature, value in st.session_state.get('record_inputs', {}).items():
        st.session_state.setdefault(f'field_{feature}', value)
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
        st.session_state["record_inputs"] = {name: None for name in FEATURES}
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
    st.session_state["record_inputs"] = values.copy()
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


def render_batch():
    ctx = context()
    ROOT, REPORTS = ctx['root'], ctx['reports']
    run, model, metadata = ctx['run'], ctx['model'], ctx['metadata']
    header('📁 CSV batch prediction', 'Predict dataset labels for multiple numerical records.')
    quality_notice()
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
