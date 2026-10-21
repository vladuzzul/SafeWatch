"""Recording-level held-out evaluation for the selected architecture."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .metrics import classification_metrics


def write_report(output, results, predictions, complete):
    pd.DataFrame(results).to_csv(output / "fold_metrics.csv", index=False)
    frame = pd.concat(predictions, ignore_index=True)
    frame.to_csv(output / "out_of_fold_predictions.csv", index=False)
    metrics = classification_metrics(frame.label, frame.score, .5)
    # Each fold keeps its validation-selected threshold for binary decisions.
    tn, fp, fn, tp = confusion_matrix(frame.label, frame.prediction, labels=[0, 1]).ravel()
    metrics.update({"accuracy": float((tn + tp) / len(frame)),
                    "sensitivity": float(tp / (tp + fn)),
                    "specificity": float(tn / (tn + fp)),
                    "balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2),
                    "precision": float(tp / (tp + fp)) if tp + fp else 0.,
                    "f1": float(2 * tp / (2 * tp + fp + fn)),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    summary = {"model": "inceptiontime", "recordings": len(frame),
               "folds": len(results), "complete_cross_validation": complete, **metrics}
    for key in ("accuracy", "sensitivity", "specificity", "f1", "roc_auc", "average_precision"):
        values = [row[key] for row in results]
        summary[f"{key}_fold_mean"] = float(np.mean(values))
        summary[f"{key}_fold_std"] = float(np.std(values))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)
    status = "Full evaluation" if complete else "Quick pipeline check"
    report = f"""# InceptionTime: {status}

- Recordings tested: {len(frame)}
- Accuracy: {metrics['accuracy']:.4%}
- Sensitivity (falls found): {metrics['sensitivity']:.4%}
- Specificity (non-falls correctly identified): {metrics['specificity']:.4%}
- F1: {metrics['f1']:.4f}
- Missed fall recordings: {fn}; false alarms: {fp}.

The training stop point and decision threshold are chosen using validation data only.
Training, validation and test data come from different people in each fold.
These results describe whole recordings. The highest window score represents each recording.
Pooled ROC-AUC and average precision combine scores from separate models. Also check the fold averages.
A quick pipeline check does not replace the full five-fold evaluation.
These results do not measure detection delay or false alarms per hour on a continuous signal.
"""
    (output / "report.md").write_text(report)
    return summary
