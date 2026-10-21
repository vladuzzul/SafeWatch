"""Check scientific split invariants, reusable inference and dataset preparation."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import savemat

from safewatch_ml.data import (CHANNELS, Recording, RecordingBags, collate_bags,
                               fit_normalizer, make_windows, read_signal, subject_splits)
from safewatch_ml.inference import FallDetector
from safewatch_ml.metrics import classification_metrics, select_threshold
from safewatch_ml.models import InceptionTime, bag_logits
from safewatch_ml.paths import project_root, default_checkpoint
from safewatch_ml.prepare import SENSOR_COLUMNS, convert_file, create_index
from safewatch_ml.train import validate_config


torch.set_num_threads(2)


def write_recording(path):
    frame = pd.DataFrame({'year': [2020] * 7, 'month': [1] * 7, 'day': [1] * 7,
                          'hour': [0] * 7, 'minute': [0] * 7,
                          'second': [0, .02, .02, .04, .06, .08, .10]})
    for channel in CHANNELS:
        frame[channel] = [0, 1, 3, 4, 6, 8, 10]
    frame.to_csv(path, index=False)


def test_subject_splits_are_disjoint_and_cover_each_recording_once():
    records = [Recording(f'{subject}/{label}', f'subject_{subject:02}', label,
                         np.full((10, 6), subject, dtype=np.float32), {})
               for subject in range(1, 22) for label in (0, 1)]
    tested = []
    for _, train, val, test in subject_splits(records, 5, 42, .25):
        groups = [{records[int(i)].subject for i in part} for part in (train, val, test)]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        tested.extend(test)
    assert sorted(tested) == list(range(len(records)))


def test_normalizer_never_uses_validation_or_test():
    records = [Recording('a', 's1', 0, np.ones((4, 6), dtype=np.float32), {}),
               Recording('b', 's2', 1, np.full((4, 6), 999, dtype=np.float32), {})]
    mean, std = fit_normalizer(records, [0])
    np.testing.assert_allclose(mean, 1)
    assert np.isfinite(std).all() and (std > 0).all()


def test_duplicate_timestamps_are_averaged_and_backwards_times_rejected(tmp_path):
    path = tmp_path / 'signal.csv'
    write_recording(path)
    signal, audit = read_signal(path, 50)
    assert signal.shape == (6, 6)
    np.testing.assert_allclose(signal[:, 0], [0, 2, 4, 6, 8, 10])
    assert audit['duplicate_timestamps'] == 1
    frame = pd.read_csv(path)
    frame.loc[3, 'second'] = .01
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='nondecreasing'):
        read_signal(path, 50)


def test_window_tail_and_short_recording():
    signal = np.arange(66, dtype=np.float32).reshape(11, 6)
    windows = make_windows(signal, 6, 3)
    np.testing.assert_array_equal(windows[-1], signal[-6:].T)
    short = make_windows(signal[:2], 6, 3)
    assert short.shape == (1, 6, 6)
    np.testing.assert_array_equal(short[0, :, -1], signal[1])


def test_recording_targets_and_mil_gradient_semantics():
    records = [Recording('a', 's1', 1, np.zeros((20, 6), np.float32), {}),
               Recording('b', 's2', 0, np.zeros((12, 6), np.float32), {})]
    dataset = RecordingBags(records, [0, 1], np.zeros(6), np.ones(6), 8, 4)
    windows, owners, labels, ids = collate_bags([dataset[0], dataset[1]])
    assert len(labels) == 2 and len(windows) > 2 and len(windows) == len(owners)
    assert labels.tolist() == [1, 0] and ids.tolist() == [0, 1]
    logits = torch.tensor([-2., 3., -4.], requires_grad=True)
    pooled = bag_logits(logits, torch.tensor([0, 0, 1]), 2)
    assert pooled.tolist() == [3., -4.]
    pooled.sum().backward()
    assert logits.grad.tolist() == [0., 1., 1.]


def test_selected_model_has_finite_gradients_for_variable_window_lengths():
    model = InceptionTime()
    for length in [100, 200, 300]:
        output = model(torch.randn(3, 6, length))
        assert output.shape == (3,) and torch.isfinite(output).all()
        output.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)


def test_threshold_constraint_and_tied_scores():
    y = np.array([0, 0, 1, 1]); scores = np.array([.1, .8, .7, .9])
    threshold = select_threshold(y, scores, 1.)
    metrics = classification_metrics(y, scores, threshold)
    assert metrics['specificity'] == 1 and metrics['sensitivity'] == .5
    threshold = select_threshold(y, np.array([.5] * 4), 1.)
    assert threshold > .5


def test_checkpoint_round_trip_public_inference_and_contract(tmp_path):
    path = tmp_path / 'signal.csv'; write_recording(path)
    config = {'sample_rate': 50, 'window_seconds': 1, 'stride_seconds': .5,
              'inception_width': 16, 'inception_depth': 6, 'inception_bottleneck': 16,
              'inception_kernels': [9, 19, 39]}
    torch.manual_seed(42); model = InceptionTime().eval()
    checkpoint = {'format_version': 1, 'channels': list(CHANNELS), 'architecture': 'inceptiontime',
                  'config': config, 'normalizer_mean': [0.] * 6, 'normalizer_std': [1.] * 6,
                  'threshold': .5, 'state_dict': model.state_dict(),
                  'label_scope': 'recording', 'aggregation': 'max_window_logit'}
    torch.save(checkpoint, tmp_path / 'model.pt')
    detector = FallDetector(tmp_path / 'model.pt')
    result = detector.predict_csv(path)
    signal, _ = read_signal(path, 50)
    direct = detector.predict_signal(signal, sample_rate=50)
    assert result['score'] == direct['score']
    with torch.inference_mode():
        windows = torch.from_numpy(make_windows(signal, 50, 25))
        expected = float(model(windows).max().sigmoid())
    assert result['score'] == pytest.approx(expected)
    contract = json.loads((project_root() / 'shared/contracts/fall-classification.schema.json').read_text())
    assert set(contract['required']) <= result.keys()
    assert set(result) <= contract['properties'].keys()
    assert result['prediction'] == int(result['score'] >= result['threshold'])
    with pytest.raises(ValueError, match='uniform sampling'):
        detector.predict_signal(signal, sample_rate=100)
    with pytest.raises(ValueError, match='nonempty'):
        detector.predict_signal(np.zeros((5, 3)), sample_rate=50)


def test_paths_work_from_a_different_directory(tmp_path, monkeypatch):
    root = project_root(); monkeypatch.chdir(tmp_path)
    assert project_root() == root
    assert default_checkpoint() == root / 'artifacts/models/inceptiontime/cv/fold_1/checkpoint.pt'


def test_inference_does_not_import_training_services():
    subprocess.run([sys.executable, '-c',
        "import sys; from safewatch_ml.inference import FallDetector; "
        "assert 'mlflow' not in sys.modules; assert 'sklearn' not in sys.modules"], check=True)


def test_mat_conversion_and_relocatable_index(tmp_path):
    source = tmp_path / 'raw.mat'
    time = np.array([[2020, 1, 1, 0, 0, 0], [2020, 1, 1, 0, 0, .02]])
    fields = {name: np.ones((2, 1)) for name in SENSOR_COLUMNS}; fields['time'] = time
    savemat(source, fields)
    output = tmp_path / 'processed/subject_01/fall/fall1.csv'
    assert convert_file(source, output) == 2
    index = tmp_path / 'index.csv'
    assert create_index(tmp_path / 'processed', index) == 1
    frame = pd.read_csv(index)
    assert frame.iloc[0].path == 'processed/subject_01/fall/fall1.csv'
    assert frame.iloc[0].label == 1 and (index.parent / frame.iloc[0].path).is_file()


def test_active_config_has_one_seed_and_one_architecture():
    config = json.loads((project_root() / 'ml/configs/inceptiontime.json').read_text())
    validate_config(config)
    assert config['seed'] == 11 and 'seeds' not in config
