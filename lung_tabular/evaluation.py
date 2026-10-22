"""Presentation helpers using saved evaluation evidence, never refitting models."""
import pandas as pd

MODEL_NAMES = {
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "logistic_C0.1": "Logistic Regression (C=0.1)",
    "logistic_C1": "Logistic Regression (C=1)",
    "random_forest_depth6": "Random Forest (depth 6)",
    "random_forest_full": "Random Forest (full depth)",
}


def model_name(name):
    return MODEL_NAMES.get(name, name.replace("_", " ").title())


def metric_table(evaluation, focus="YES class"):
    key = "macro avg" if focus == "Macro average" else "YES"
    rows = []
    labels = [model_name(evaluation["model"]), "Majority-label baseline"]
    for result in (evaluation, evaluation["baseline"]):
        scores = result["per_class"][key]
        rows.append({"Accuracy": result["accuracy"], "Precision": scores["precision"],
                     "Recall": scores["recall"], "F1": scores["f1-score"],
                     "Balanced accuracy": result["balanced_accuracy"],
                     "Macro F1": result["macro_f1"], "ROC AUC": result["roc_auc"]})
    return pd.DataFrame(rows, index=pd.Index(labels, name="Model"))


def cv_table(comparison):
    frame = pd.DataFrame(comparison).copy()
    frame["model"] = frame["model"].map(model_name)
    names = {"model": "Model", "cv_macro_f1": "CV macro F1", "cv_macro_f1_std": "F1 std",
             "cv_balanced_accuracy": "CV balanced accuracy", "cv_balanced_accuracy_std": "Balanced accuracy std",
             "cv_roc_auc": "CV ROC AUC", "cv_roc_auc_std": "ROC AUC std"}
    return frame.rename(columns=names).set_index("Model")
