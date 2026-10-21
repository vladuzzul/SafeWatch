# Fall detection with InceptionTime

This folder contains the ML code for fall classification. It uses one
InceptionTime model in PyTorch. The settings are in
[configs/inceptiontime.json](configs/inceptiontime.json).

The current model classifies complete sensor recordings. Work on real-time
alerts is still needed.

## Setup

Run the commands below from the project root, the folder that contains
`Makefile` and `ml/`.

Use Python 3.11 or newer. Create a virtual environment if one does not exist:

```bash
python3 -m venv .venv
```

Install the package and the tools used for training and tests:

```bash
make setup
```

This runs `pip install -e './ml[train,dev]'`. The `-e` option means that changes
to the source code are used without installing the package again. The `train`
and `dev` groups add tools for training and testing.

`requirements-lock.txt` lists the package versions from the first experiment.
To use those versions, install that file before installing the local package:

```bash
.venv/bin/python -m pip install -r ml/requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e ./ml
```

## Data

The model uses the HIFD dataset. See the [data guide](../data/README.md) for the
source and local folder layout.

Each recording has one label:

- `1`: the recording contains a fall.
- `0`: the recording does not contain a fall.

Prepare the CSV files and index with:

```bash
make prepare-data
```

Training uses six channels, in this order:

```text
ax, ay, az, droll, dpitch, dyaw
```

The first three describe acceleration. The last three describe rotation speed.
The model does not use the heart signal or orientation values.

The data is sampled at 50 Hz: 50 measurements per second. When several rows
have the same timestamp, their values are averaged. The values are then placed
on a regular time grid. The dataset acceleration already has gravity removed.
Data from another device must use matching units and sensor settings.

## How training works

A recording is split into six-second windows. Each window contains 300
measurements. Windows start every three seconds during training and every
0.6 seconds during validation and testing.

A fall recording may also contain rest before or after the fall. To avoid
marking all of those windows as falls, the model gives each window a score
and uses the highest score for the whole recording. Training compares that
result with the recording label. This method is called multiple instance
learning.

The data is divided into five groups, called folds. Each group is used once
for testing. Within each fold, training, validation and test recordings come
from different people:

- Training data teaches the model.
- Validation data chooses when to stop training and where to set the decision threshold.
- Test data checks the saved model on people it did not learn from.

The mean and standard deviation used to scale the inputs are calculated from
training data only. The threshold aims to find as many falls as possible while
correctly identifying at least 95% of non-fall validation recordings. The test
results can be different.

Start the full evaluation with:

```bash
make train
```

For a quick check of the training code, run:

```bash
make smoke
```

The quick check runs one epoch, which is one pass through the training data,
and one fold. Its results are marked as a partial evaluation.

New runs are saved in `artifacts/runs/`. A full run saves five separate models
of the same type, one for each fold. They are used separately.

Each saved model file, called a checkpoint, contains the learned weights,
settings, input scaling values, decision threshold and participant groups.

## Use a saved model

You can use the saved weights without training again:

```bash
make predict
```

By default, this uses the first fold checkpoint and an example recording.
You can choose other files:

```bash
make predict CHECKPOINT=artifacts/models/inceptiontime/cv/fold_1/checkpoint.pt RECORDING=data/hifd/processed/subject_07/fall/fall1.csv
```

To use the model from Python:

```python
from safewatch_ml.inference import FallDetector

detector = FallDetector("artifacts/models/inceptiontime/cv/fold_1/checkpoint.pt")
result = detector.predict_csv("data/hifd/processed/subject_07/fall/fall1.csv")
```

Load the detector once and reuse it for more recordings. This class does not
import MLflow or scikit-learn.

For a NumPy array, use `detector.predict_signal(signal, sample_rate=50)`.
The array must have shape `[number_of_samples, 6]`, with the channel order
listed above and regular sampling at 50 Hz.

The result includes:

| Field | Meaning |
|---|---|
| `score` | The model score between 0 and 1; it is not a verified probability |
| `threshold` | The cutoff used to make the decision |
| `prediction` | `1` if the score reaches the threshold; otherwise `0` |
| `label_scope` | `recording`, because the decision describes the whole recording |
| `window_scores` | The score of each window |
| `schema_version` | The result format version |

The result format is defined in
[fall-classification.schema.json](../shared/contracts/fall-classification.schema.json).

The five original InceptionTime checkpoints are stored in
`artifacts/models/inceptiontime/cv/fold_1/` through `fold_5/`. The first fold
is a fixed example, not a final model chosen for its test score. Data and
weights are kept outside Git and must be restored on a new computer.

## Track runs with MLflow

```bash
make mlflow
```

Open [MLflow](http://127.0.0.1:5000). New runs use the
`SafeWatch-InceptionTime` experiment. Earlier experiments remain in the history.

MLflow saves the settings, scores over training, data splits and model files.
Its database is `artifacts/mlflow/mlflow.db`. Its saved files are in
`artifacts/mlflow/artifacts/`.

To use another port, run `make mlflow PORT=5050`.

## Tests

```bash
make test
```

The tests use small, generated signals. They do not need the local dataset
or trained weights.

## Main source files

| File in `src/safewatch_ml/` | Purpose |
|---|---|
| `models.py` | Defines InceptionTime |
| `prepare.py` | Converts MAT files to CSV and creates the index |
| `data.py` | Reads recordings, creates windows and splits people into groups |
| `train.py` | Trains the model and logs results in MLflow |
| `metrics.py` and `evaluate.py` | Choose thresholds, calculate scores and write reports |
| `inference.py` | Loads a saved model and makes predictions |
| `predict.py` | Runs a prediction from the terminal |
| `tracking.py` | Starts the local MLflow server |

See the [model results and limits](../docs/ml/model-card.md) and
[research sources](../docs/research/README.md) for more detail.
