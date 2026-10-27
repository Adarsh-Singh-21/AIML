"""Compatibility command for the diagnostics defined in notebooks/train.ipynb."""
from . import DATA, REPORTS


def generate_diagnostics(data_dir=DATA, reports=REPORTS):
    from .notebook_runner import run_workflow
    return run_workflow(data_dir=data_dir, reports=reports, diagnostics=True)


if __name__ == "__main__":
    print(generate_diagnostics())
