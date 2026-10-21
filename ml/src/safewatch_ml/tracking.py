"""Start the local MLflow UI using the repository's managed storage paths."""

import argparse
import os
import subprocess
import sys

from .paths import project_root, tracking_uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    storage = project_root() / "artifacts/mlflow"
    (storage / "artifacts").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
    raise SystemExit(subprocess.call([
        sys.executable, "-m", "mlflow", "server", "--backend-store-uri", tracking_uri(),
        "--default-artifact-root", (storage / "artifacts").as_uri(),
        "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1",
    ], env=env))


if __name__ == "__main__":
    main()
