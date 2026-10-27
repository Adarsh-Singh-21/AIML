"""Compatibility command for the diagnostics defined in notebooks/train.ipynb."""
from . import DATA, REPORTS


def generate_diagnostics(data_dir=DATA, reports=REPORTS):
    from .notebook_workflow import workflow_function
    return workflow_function("generate_diagnostics")(data_dir, reports)


if __name__ == "__main__":
    print(generate_diagnostics())
