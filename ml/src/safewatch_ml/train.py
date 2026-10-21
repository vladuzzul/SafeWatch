"""Train and evaluate the selected InceptionTime model on disjoint subjects."""

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (CHANNELS, RecordingBags, collate_bags, fit_normalizer,
                   load_recordings, subject_splits)
from .metrics import classification_metrics, select_threshold
from .models import bag_logits, build_model


from .paths import project_root, tracking_uri
from .evaluate import write_report


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                          + "\n", encoding="utf-8")


def resolve_device(request):
    if request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Exact determinism can depend on device and torch version. Fail on known
    # nondeterminism except for backends that only offer best-effort support.
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset, batch_size, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_bags, num_workers=0, generator=generator)


@torch.inference_mode()
def predict(model, loader, device, loss_function=None):
    model.eval()
    labels, scores, indices = [], [], []
    loss_sum = 0.0
    for windows, owners, target, index in loader:
        windows, target = windows.to(device), target.to(device)
        # Chunk evaluation to keep the largest negative recording affordable.
        window_logits = torch.cat([model(chunk) for chunk in windows.split(128)])
        logits = bag_logits(window_logits, owners, len(target))
        if loss_function is not None:
            loss_sum += float(loss_function(logits, target).item()) * len(target)
        labels.extend(target.cpu().tolist())
        scores.extend(logits.sigmoid().cpu().tolist())
        indices.extend(index.tolist())
    return (np.asarray(labels, dtype=int), np.asarray(scores),
            np.asarray(indices), loss_sum / len(labels))


def train_fold(seed, config, recordings, parts, normalizer,
                 output, device, fingerprint):
    set_seed(seed)
    train_indices, val_indices, test_indices = parts
    mean, std = normalizer
    window_samples = round(config["sample_rate"] * config["window_seconds"])
    # Reduce redundant training windows; evaluate densely and identically for
    # evaluation. This is not a random subset selected using labels.
    bags = [RecordingBags(recordings, indices, mean, std, window_samples,
                           round(config["sample_rate"] * stride))
            for indices, stride in zip(parts, (config["train_stride_seconds"],
                                                config["stride_seconds"],
                                                config["stride_seconds"]))]
    loaders = [make_loader(bags[0], config["batch_size"], True, seed),
               make_loader(bags[1], config["batch_size"]),
               make_loader(bags[2], config["batch_size"])]
    model = build_model(config).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    labels = np.array([recordings[int(i)].label for i in train_indices])
    positive_weight = float((labels == 0).sum() / (labels == 1).sum())
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight,
                                                               device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                   weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                          factor=0.5, patience=3)
    output.mkdir(parents=True, exist_ok=False)
    best_key = (-float("inf"), -float("inf"))
    best_state, best_epoch, stale = None, 0, 0
    history = []
    started = time.perf_counter()
    mlflow.log_params({"architecture": "inceptiontime", "seed": seed,
                       "trainable_parameters": parameters,
                       "positive_weight": positive_weight,
                       "train_recordings": len(train_indices),
                       "validation_recordings": len(val_indices),
                       "test_recordings": len(test_indices)})
    for epoch in range(config["epochs"]):
        model.train()
        running = 0.0
        epoch_start = time.perf_counter()
        for windows, owners, target, _ in loaders[0]:
            windows, target = windows.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = bag_logits(model(windows), owners, len(target))
            loss = loss_function(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in InceptionTime, seed {seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item()) * len(target)
        val_y, val_scores, _, val_loss = predict(model, loaders[1], device, loss_function)
        validation = classification_metrics(val_y, val_scores, 0.5)
        row = {"epoch": epoch + 1, "train_loss": running / len(bags[0]),
               "validation_loss": val_loss,
               "validation_average_precision": validation["average_precision"],
               "validation_roc_auc": validation["roc_auc"],
               "learning_rate": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
        key = (validation["average_precision"], -val_loss)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step(validation["average_precision"])
        print(f"  inceptiontime seed={seed} epoch={epoch + 1}/{config['epochs']} "
              f"loss={row['train_loss']:.4f} val_AP={key[0]:.4f} "
              f"({row['epoch_seconds']:.1f}s)", flush=True)
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    val_y, val_scores, val_order, _ = predict(model, loaders[1], device)
    threshold = select_threshold(val_y, val_scores, config["minimum_specificity"])
    test_y, test_scores, test_order, _ = predict(model, loaders[2], device)
    metrics = classification_metrics(test_y, test_scores, threshold)
    metrics_at_half = classification_metrics(test_y, test_scores, 0.5)
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    mlflow.log_metrics({f"test_at_0_5_{k}": v for k, v in metrics_at_half.items()})
    mlflow.log_metrics({"best_epoch": best_epoch, "threshold": threshold,
                        "training_seconds": time.perf_counter() - started})
    mlflow.log_metrics({f"validation_{k}": v for k, v in
                       classification_metrics(val_y, val_scores, threshold).items()})
    checkpoint = {
        "format_version": 1, "architecture": "inceptiontime", "seed": seed,
        "state_dict": best_state, "config": config,
        "normalizer_mean": mean.tolist(), "normalizer_std": std.tolist(),
        "channels": list(CHANNELS), "threshold": threshold,
        "aggregation": "max_window_logit", "label_scope": "recording",
        "dataset_sha256": fingerprint, "best_epoch": best_epoch,
        "train_subjects": sorted({recordings[int(i)].subject for i in train_indices}),
        "validation_subjects": sorted({recordings[int(i)].subject for i in val_indices}),
        "test_subjects": sorted({recordings[int(i)].subject for i in test_indices}),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    save_json(output / "metrics.json", {"threshold": threshold, "test": metrics,
                                        "test_at_0_5": metrics_at_half})
    for split, ids, truth, probabilities in (
        ("validation", val_order, val_y, val_scores),
        ("test", test_order, test_y, test_scores),
    ):
        prediction_frame(recordings, ids, truth, probabilities, threshold).to_csv(
            output / f"{split}_predictions.csv", index=False)
    mlflow.log_artifacts(str(output), artifact_path="model")
    return {"val_y": val_y, "val_scores": val_scores, "val_order": val_order,
            "test_y": test_y, "test_scores": test_scores, "test_order": test_order,
            "threshold": threshold, "metrics": metrics,
            "parameters": parameters, "checkpoint": output / "checkpoint.pt"}


def prediction_frame(recordings, ids, truth, probabilities, threshold):
    return pd.DataFrame({
        "recording_index": ids, "path": [recordings[int(i)].path for i in ids],
        "subject": [recordings[int(i)].subject for i in ids],
        "label": truth, "score": probabilities, "threshold": threshold,
        "prediction": (probabilities >= threshold).astype(int),
    })


def validate_config(config):
    for key in ('sample_rate', 'window_seconds', 'stride_seconds', 'train_stride_seconds',
                'epochs', 'patience', 'batch_size', 'learning_rate', 'cpu_threads',
                'inception_width', 'inception_bottleneck'):
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if config['folds'] < 2 or not 0 < config['validation_fraction'] < 1:
        raise ValueError('Invalid folds or validation fraction')
    if not 0 <= config['minimum_specificity'] <= 1 or config['weight_decay'] < 0:
        raise ValueError('Invalid specificity or weight decay')
    for key in ('stride_seconds', 'train_stride_seconds'):
        if config[key] > config['window_seconds'] or round(config['sample_rate'] * config[key]) < 1:
            raise ValueError('Stride must be at least one sample and at most the window length')
    if round(config['sample_rate'] * config['window_seconds']) < 4:
        raise ValueError('Window must contain at least four samples')
    if not isinstance(config['seed'], int):
        raise ValueError('seed must be an integer')
    build_model(config)


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'ml/configs/inceptiontime.json')
    parser.add_argument('--index', type=Path, default=root / 'data/hifd/index.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tracking-uri', default=tracking_uri())
    parser.add_argument('--fold-limit', type=int, help='Partial pipeline check only')
    parser.add_argument('--epochs', type=int, help='Override the maximum number of epochs')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    config = copy.deepcopy(json.loads(args.config.read_text()))
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.device is not None:
        config['device'] = args.device
    validate_config(config)
    if args.fold_limit is not None and not 1 <= args.fold_limit <= config['folds']:
        parser.error('fold-limit must lie between 1 and folds')
    config['fold_limit'] = args.fold_limit
    output = (args.output or root / 'artifacts/runs' /
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (root / 'artifacts/mlflow').mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['cpu_threads'])
    device = resolve_device(config['device'])
    print(f'Device: {device}. Loading recordings...', flush=True)
    recordings, fingerprint = load_recordings(args.index.resolve(), config['sample_rate'])
    splits = list(subject_splits(recordings, config['folds'], config['split_seed'],
                                 config['validation_fraction']))
    if args.fold_limit:
        splits = splits[:args.fold_limit]
    complete = len(splits) == config['folds']
    save_json(output / 'config.json', config)
    pd.DataFrame([{'path': r.path, 'subject': r.subject, 'label': r.label, **r.audit}
                  for r in recordings]).to_csv(output / 'data_audit.csv', index=False)
    save_json(output / 'environment.json', {
        'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
        'mlflow': mlflow.__version__, 'device': str(device), 'dataset_sha256': fingerprint,
        'channels': list(CHANNELS), 'label_scope': 'recording',
        'complete_cross_validation': complete,
    })
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                             text=True, check=True).stdout
    (output / 'requirements-lock.txt').write_text(freeze)
    with zipfile.ZipFile(output / 'source.zip', 'w') as archive:
        for path in sorted(Path(__file__).parent.glob('*.py')):
            archive.write(path, f'safewatch_ml/{path.name}')
    os.environ.setdefault('MLFLOW_ENABLE_ASYNC_LOGGING', 'false')
    os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(config['experiment_name'])
    if experiment is None:
        artifacts = root / 'artifacts/mlflow/artifacts'
        artifacts.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(config['experiment_name'],
                                                  artifact_location=artifacts.as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    results, predictions = [], []
    with mlflow.start_run(run_name=output.name) as parent:
        save_json(output / 'mlflow.json', {'run_id': parent.info.run_id,
                                           'experiment_id': experiment_id,
                                           'tracking_uri': args.tracking_uri})
        mlflow.log_params({k: json.dumps(v) if isinstance(v, list) else v
                           for k, v in config.items()})
        mlflow.set_tags({'architecture': 'inceptiontime', 'label_scope': 'recording',
                         'split_unit': 'subject', 'dataset_sha256': fingerprint,
                         'complete_cross_validation': str(complete), 'device': str(device)})
        for filename in ('config.json', 'environment.json', 'data_audit.csv',
                         'requirements-lock.txt', 'source.zip', 'mlflow.json'):
            mlflow.log_artifact(str(output / filename))
        for fold, train, val, test in splits:
            fold_output = output / f'fold_{fold + 1}'
            fold_output.mkdir()
            parts = (train, val, test)
            pd.DataFrame([
                {'split': name, 'path': recordings[int(i)].path,
                 'subject': recordings[int(i)].subject, 'label': recordings[int(i)].label}
                for name, indices in zip(('train', 'validation', 'test'), parts)
                for i in indices
            ]).to_csv(fold_output / 'split.csv', index=False)
            normalizer = fit_normalizer(recordings, train)
            print(f'Fold {fold + 1}/{config["folds"]}: train={len(train)}, '
                  f'validation={len(val)}, test={len(test)}', flush=True)
            with mlflow.start_run(run_name=f'fold_{fold + 1}', nested=True) as run:
                mlflow.log_params({'fold': fold + 1, 'config_parent_run_id': parent.info.run_id})
                mlflow.log_artifact(str(fold_output / 'split.csv'))
                result = train_fold(config['seed'], config, recordings, parts, normalizer,
                                    fold_output / 'model', device, fingerprint)
                save_json(fold_output / 'mlflow.json', {'run_id': run.info.run_id})
            results.append({'fold': fold + 1, 'model': 'inceptiontime',
                             'parameters': result['parameters'], 'threshold': result['threshold'],
                             **result['metrics']})
            predictions.append(prediction_frame(recordings, result['test_order'], result['test_y'],
                                                 result['test_scores'], result['threshold']).assign(
                                                     fold=fold + 1, model='inceptiontime'))
            save_json(output / 'progress.json', {'completed_folds': len(results),
                                                 'requested_folds': len(splits), 'status': 'running'})
            pd.DataFrame(results).to_csv(output / 'fold_metrics.csv', index=False)
        summary = write_report(output, results, predictions, complete)
        if complete:
            ids = pd.concat(predictions).recording_index
            if len(ids) != len(recordings) or ids.nunique() != len(recordings):
                raise RuntimeError('Each recording must appear exactly once in held-out evaluation')
        save_json(output / 'progress.json', {'completed_folds': len(splits),
                                             'requested_folds': len(splits), 'status': 'finished',
                                             'complete_cross_validation': complete})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        for filename in ('fold_metrics.csv', 'out_of_fold_predictions.csv', 'summary.csv',
                         'summary.json', 'report.md', 'progress.json'):
            mlflow.log_artifact(str(output / filename), artifact_path='evaluation')
    print(f'Finished. Results: {output}', flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
