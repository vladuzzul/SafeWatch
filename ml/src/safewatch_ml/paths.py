"""Storage paths for the repository, independent of the current directory."""

import os
from pathlib import Path


def project_root():
    override = os.environ.get("SAFEWATCH_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"SAFEWATCH_ROOT does not exist: {root}")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "ml/pyproject.toml").is_file():
            return root
    raise RuntimeError("Set SAFEWATCH_ROOT to your SafeWatch checkout")


def tracking_uri():
    return f"sqlite:///{project_root() / 'artifacts/mlflow/mlflow.db'}"


def default_checkpoint():
    # A fixed CV reference for development, never selected by highest test score.
    return project_root() / "artifacts/models/inceptiontime/cv/fold_1/checkpoint.pt"
