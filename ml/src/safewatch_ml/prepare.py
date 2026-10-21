"""Convert raw HIFD MAT recordings and create the processed CSV index."""
import argparse
import csv
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from .paths import project_root


TIME_COLUMNS = ("year", "month", "day", "hour", "minute", "second")
SENSOR_COLUMNS = (
    "w", "x", "y", "z", "droll", "dpitch", "dyaw", "ax", "ay", "az", "heart"
)


def convert_file(source: Path, destination: Path) -> int:
    """Write one CSV with a row per measurement; return its row count."""
    mat = loadmat(source)
    missing = [name for name in ("time", *SENSOR_COLUMNS) if name not in mat]
    if missing:
        raise ValueError(f"{source}: missing fields: {', '.join(missing)}")

    time = mat["time"]
    if time.ndim != 2 or time.shape[1] != len(TIME_COLUMNS):
        raise ValueError(f"{source}: expected time with 6 columns, got {time.shape}")

    columns = {name: time[:, index] for index, name in enumerate(TIME_COLUMNS)}
    for name in SENSOR_COLUMNS:
        values = mat[name]
        if values.shape != (len(time), 1):
            raise ValueError(
                f"{source}: expected {name} with shape ({len(time)}, 1), "
                f"got {values.shape}"
            )
        columns[name] = values[:, 0]

    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(destination, index=False)
    return len(time)


def create_index(processed: Path, index_path: Path) -> int:
    sources = sorted(processed.rglob('*.csv'))
    if not sources:
        raise FileNotFoundError(f'No CSV recordings found: {processed}')
    labels = {'fall': 1, 'non-fall': 0}
    rows = []
    for source in sources:
        if source.parent.name not in labels:
            raise ValueError(f'Unknown label category: {source}')
        rows.append((source.relative_to(index_path.parent).as_posix(), labels[source.parent.name]))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(('path', 'label'))
        writer.writerows(rows)
    return len(rows)


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=root / 'data/hifd/raw')
    parser.add_argument('--processed', type=Path, default=root / 'data/hifd/processed')
    parser.add_argument('--index', type=Path, default=root / 'data/hifd/index.csv')
    args = parser.parse_args()
    sources = sorted(args.raw.rglob('*.mat'))
    if not sources:
        raise FileNotFoundError(f'No MAT recordings found: {args.raw}')
    total = 0
    for source in sources:
        destination = args.processed / source.relative_to(args.raw).with_suffix('.csv')
        total += convert_file(source, destination)
    count = create_index(args.processed, args.index)
    print(f'Converted {count} recordings ({total} rows). Index: {args.index}')


if __name__ == '__main__':
    main()
