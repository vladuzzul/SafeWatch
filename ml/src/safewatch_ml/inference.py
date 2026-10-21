"""Reusable inference API: no MLflow or training dependencies are imported."""

from pathlib import Path

import numpy as np
import torch

from .data import CHANNELS, make_windows, read_signal
from .models import build_model


class FallDetector:
    """Load once and reuse for complete, offline recording classification.

    Array inputs must be uniformly sampled in the documented channel order,
    with the same units and gravity treatment as the training dataset.
    """

    def __init__(self, checkpoint_path: str | Path, device="cpu"):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}. Restore local artifacts or supply a checkpoint.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("channels") != list(CHANNELS):
            raise ValueError("Unsupported checkpoint format or channel order")
        if checkpoint.get("architecture") not in {"inception", "inceptiontime"}:
            raise ValueError("Only the selected InceptionTime architecture is supported")
        if checkpoint.get("label_scope") != "recording" or checkpoint.get("aggregation") != "max_window_logit":
            raise ValueError("Expected a recording-level max-window-logit checkpoint")
        self.config = checkpoint["config"]
        self.device = torch.device(device)
        self.threshold = float(checkpoint["threshold"])
        self.mean = np.asarray(checkpoint["normalizer_mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["normalizer_std"], dtype=np.float32)
        if (self.mean.shape != (6,) or self.std.shape != (6,) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all() or np.any(self.std <= 0)
                or not np.isfinite(self.threshold)):
            raise ValueError("Invalid normalizer or threshold in checkpoint")
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict_signal(self, signal, *, sample_rate):
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != len(CHANNELS) or not len(signal):
            raise ValueError("Expected nonempty [samples, 6] input in the documented channel order")
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains non-finite values")
        if sample_rate != self.config["sample_rate"]:
            raise ValueError(f"Expected uniform sampling at {self.config['sample_rate']} Hz")
        signal = (signal - self.mean) / self.std
        windows = make_windows(signal, round(sample_rate * self.config["window_seconds"]),
                               round(sample_rate * self.config["stride_seconds"]))
        logits = torch.cat([self.model(chunk.to(self.device)) for chunk in
                            torch.from_numpy(windows).split(128)])
        score = float(logits.max().sigmoid().cpu())
        return {"schema_version": "1.0", "model": "inceptiontime", "score": score,
                "threshold": self.threshold, "prediction": int(score >= self.threshold),
                "window_scores": logits.sigmoid().cpu().tolist(), "label_scope": "recording"}

    def predict_csv(self, path: str | Path):
        signal, audit = read_signal(Path(path), self.config["sample_rate"])
        result = self.predict_signal(signal, sample_rate=self.config["sample_rate"])
        result["audit"] = audit
        return result
