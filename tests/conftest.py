import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    folder = Path(__file__).resolve().parents[1] / ".cache/test_tmp"
    folder.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(folder)


@pytest.fixture
def evaluation_evidence():
    """Saved-result-shaped fixtures for dashboard tests, independent of notebook code."""
    from sklearn.metrics import classification_report

    truth = [0, 0, 0, 1]
    selected = {
        "records": 4, "accuracy": 0.75, "balanced_accuracy": 5 / 6,
        "macro_f1": 11 / 15, "roc_auc": 2 / 3, "confusion_matrix": [[2, 1], [0, 1]],
        "per_class": classification_report(truth, [0, 0, 1, 1], labels=[0, 1],
                                            target_names=["NO", "YES"], output_dict=True, zero_division=0)
    }
    baseline = {
        "records": 4, "accuracy": 0.75, "balanced_accuracy": 0.5,
        "macro_f1": 3 / 7, "roc_auc": 0.5, "confusion_matrix": [[3, 0], [1, 0]],
        "per_class": classification_report(truth, [0, 0, 0, 0], labels=[0, 1],
                                            target_names=["NO", "YES"], output_dict=True, zero_division=0)
    }
    return {**selected, "model": "hist_gradient_boosting", "baseline": baseline}
