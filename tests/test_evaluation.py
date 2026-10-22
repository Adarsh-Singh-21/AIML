import pytest

from lung_tabular.evaluation import metric_table, cv_table
from lung_tabular.model import metrics


def test_positive_and_macro_views_do_not_mix_evaluation_types():
    evidence = metrics([0, 0, 0, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    evidence.update({"model": "hist_gradient_boosting", "baseline": metrics([0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0])})
    positive = metric_table(evidence, "YES class")
    macro = metric_table(evidence, "Macro average")
    assert len(positive) == 2
    assert positive.iloc[0]["Precision"] == pytest.approx(0.5)
    assert positive.iloc[0]["Recall"] == pytest.approx(1)
    assert macro.iloc[0]["F1"] == evidence["per_class"]["macro avg"]["f1-score"]
    assert positive["Accuracy"].equals(macro["Accuracy"])
    candidates = cv_table([{"model": "logistic_C1", "cv_macro_f1": 0.5, "cv_roc_auc": 0.52}])
    assert "CV macro F1" in candidates and "CV ROC AUC" in candidates
    assert "Accuracy" not in candidates  # A missing test metric is not invented from CV evidence.
