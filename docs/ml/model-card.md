# InceptionTime: results and limits

## Model settings

The selected model is a single InceptionTime network in PyTorch.

- Six Inception modules, each with 16 filters per branch.
- Filter lengths of 9, 19 and 39 samples, plus a max-pooling branch.
- A small layer that reduces the channel count before the longer filters.
- A shortcut connection after each group of three modules.
- A final average over time, followed by one output score.
- 108,133 learned parameters.
- Random seed 11, which fixes the starting random choices for an experiment.

The input has six IMU channels: `ax, ay, az, droll, dpitch, dyaw`.
A six-second window contains 300 measurements at 50 Hz.

## Evaluation results

The original evaluation used 349 recordings from 21 people. There were
104 recordings with a fall and 245 without a fall.

The recordings were divided into five groups for testing. In each round,
training, validation and test data came from different people. Each recording
was tested once. The training stop point and threshold were chosen using
validation data only. Seed 11 was set before checking the test results.

| Measure | Result | Meaning |
|---|---:|---|
| Accuracy | 339/349 = 97.13% | Recordings classified correctly |
| Sensitivity, also called recall | 102/104 = 98.08% | Fall recordings found |
| Specificity | 237/245 = 96.73% | Non-fall recordings correctly identified |
| Precision | 102/110 = 92.73% | Positive predictions that were correct |
| F1 | 0.9533 | A combined measure of precision and recall |
| Missed falls | 2 recordings | Fall recordings marked as non-falls |
| False alarms | 8 recordings | Non-fall recordings marked as falls |

These are combined results from the five test groups. They are not the score
of one saved model, or of a final model trained on every participant.

The result files are in
[reports/ml/inceptiontime-cv-20261021](../../reports/ml/inceptiontime-cv-20261021/).
The five local model files are in `artifacts/models/inceptiontime/cv/` at the
project root. The first fold is used as a fixed example in the commands.

## What the model predicts

The available labels describe whole recordings. They do not mark the exact
moment of a fall. The model scores each window and uses the highest score
for the recording. It learns from the recording label.

The decision threshold aims to find many falls while correctly identifying
at least 95% of non-fall recordings in validation data. This target may not
hold for new people. The output score is not a verified probability of a fall.

## Limits

The dataset contains young adults and simulated falls. The sensors were worn
on the left wrist. These results do not show performance on older adults,
real falls or another device.

The evaluation covers separate recordings. It does not measure how quickly
a live alert appears or how many false alarms occur in a full day.

The current time-grid step uses samples from the recording on both sides of
a timestamp. A live version will need a suitable buffer or another method
that works with the samples available at that time.

Before using the model on another watch, check the sensor units, axis order,
sampling rate and gravity removal. A final model also needs a separate test
on data that was not used to choose its design.

## Sources

- [InceptionTime](https://github.com/hfawaz/InceptionTime): the model design that this smaller version is based on.
- [HIFD dataset](https://github.com/nhoyh/HR_IMU_falldetection_dataset): the recordings used in training and evaluation.
- [ML guide](../../ml/README.md): setup, training and prediction commands.
