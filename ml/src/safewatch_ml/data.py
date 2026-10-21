"""Recording-level labels are bags of windows, never inherited by every window."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CHANNELS = ("ax", "ay", "az", "droll", "dpitch", "dyaw")
TIME_COLUMNS = ("year", "month", "day", "hour", "minute", "second")


@dataclass
class Recording:
    path: str
    subject: str
    label: int
    signal: np.ndarray  # [time, channels], on a uniform grid
    audit: dict


def read_signal(path: Path, sample_rate: float) -> tuple[np.ndarray, dict]:
    """Average duplicate timestamps, then interpolate each channel to a 50 Hz grid.

    Duplicate timestamps are acquisition artifacts, not duplicate training examples.
    This uses within-recording interpolation, so it is an offline preprocessing
    step; deployment must buffer samples or use a matched causal implementation.
    """
    frame = pd.read_csv(path)
    missing = set((*TIME_COLUMNS, *CHANNELS)) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    values = frame[list(CHANNELS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite IMU values")
    stamp = pd.to_datetime(frame[list(TIME_COLUMNS)], errors="raise")
    times = (stamp - stamp.iloc[0]).dt.total_seconds().to_numpy()
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError(f"{path}: timestamps must be finite and nondecreasing")
    unique, inverse = np.unique(times, return_inverse=True)
    if len(unique) < 2:
        raise ValueError(f"{path}: fewer than two distinct timestamps")
    sums = np.zeros((len(unique), len(CHANNELS)), dtype=np.float64)
    np.add.at(sums, inverse, values)
    averaged = sums / np.bincount(inverse)[:, None]
    grid = np.arange(int(np.floor(unique[-1] * sample_rate)) + 1) / sample_rate
    signal = np.stack([
        np.interp(grid, unique, averaged[:, channel])
        for channel in range(len(CHANNELS))
    ], axis=1).astype(np.float32)
    audit = {
        "raw_samples": len(frame), "resampled_samples": len(signal),
        "duration_s": float(unique[-1]),
        "duplicate_timestamps": len(times) - len(unique),
        "max_gap_s": float(np.diff(unique).max()),
    }
    return signal, audit


def load_recordings(index_path: Path, sample_rate: float):
    index = pd.read_csv(index_path)
    if set(index.columns) != {"path", "label"} or index.empty:
        raise ValueError("Index must have path,label columns and at least one recording")
    if index.path.duplicated().any() or not index.label.isin([0, 1]).all():
        raise ValueError("Index contains duplicate paths or labels outside {0,1}")
    fingerprint = sha256(index_path.read_bytes())
    recordings = []
    data_root = index_path.parent.resolve()
    for row in index.itertuples(index=False):
        relative = Path(row.path)
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root) or relative.is_absolute():
            raise ValueError(f"Recording outside dataset root: {row.path}")
        if path.parent.name != ("fall" if row.label == 1 else "non-fall"):
            raise ValueError(f"Path and label disagree: {row.path}")
        subject = path.parent.parent.name
        if not subject.startswith("subject_"):
            raise ValueError(f"Cannot identify subject in {row.path}")
        signal, audit = read_signal(path, sample_rate)
        fingerprint.update(row.path.encode())
        fingerprint.update(path.read_bytes())
        recordings.append(Recording(row.path, subject, int(row.label), signal, audit))
    return recordings, fingerprint.hexdigest()


def subject_splits(recordings, folds: int, seed: int, validation_fraction: float):
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    labels = np.array([r.label for r in recordings])
    groups = np.array([r.subject for r in recordings])
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (development, test) in enumerate(outer.split(labels, labels, groups)):
        inner = GroupShuffleSplit(n_splits=1, test_size=validation_fraction,
                                  random_state=seed + fold)
        train_local, val_local = next(inner.split(development, labels[development],
                                                  groups[development]))
        train, val = development[train_local], development[val_local]
        subject_sets = [set(groups[part]) for part in (train, val, test)]
        if any(subject_sets[a] & subject_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise RuntimeError("Subject leakage in split")
        if any(len(np.unique(labels[part])) != 2 for part in (train, val, test)):
            raise ValueError("Each split needs both classes; change folds/seed")
        yield fold, train, val, test


def fit_normalizer(recordings, train_indices):
    # One contribution per resampled measurement, not per overlapping window.
    signal = np.concatenate([recordings[int(i)].signal for i in train_indices])
    mean = signal.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = signal.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def make_windows(signal, window_samples: int, stride_samples: int):
    if window_samples < 2 or stride_samples < 1 or stride_samples > window_samples:
        raise ValueError("Require 1 <= stride <= window and window >= 2")
    if len(signal) < window_samples:
        signal = np.pad(signal, ((0, window_samples - len(signal)), (0, 0)), mode="edge")
    last = len(signal) - window_samples
    starts = list(range(0, last + 1, stride_samples))
    if starts[-1] != last:
        starts.append(last)  # Include the recording tail without dropping events.
    return np.stack([signal[start:start + window_samples].T for start in starts])


class RecordingBags(Dataset):
    def __init__(self, recordings, indices, mean, std, window_samples, stride_samples):
        self.items = []
        for index in indices:
            record = recordings[int(index)]
            windows = make_windows((record.signal - mean) / std,
                                   window_samples, stride_samples)
            self.items.append((torch.from_numpy(windows), record.label, int(index)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_bags(items):
    """Flatten only real windows. Padding must never influence BatchNorm."""
    windows, labels, indices = zip(*items)
    counts = torch.tensor([len(w) for w in windows], dtype=torch.long)
    owners = torch.repeat_interleave(torch.arange(len(items)), counts)
    return (torch.cat(windows), owners,
            torch.tensor(labels, dtype=torch.float32), torch.tensor(indices))
