"""Classify a recording with the selected InceptionTime checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from .inference import FallDetector
from .paths import default_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=default_checkpoint())
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'mps', 'cuda'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(FallDetector(args.checkpoint, args.device).predict_csv(args.recording), indent=2))


if __name__ == '__main__':
    main()
