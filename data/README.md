# ML data

The HIFD dataset is stored locally and is excluded from Git.

```text
hifd/
  raw/          Original MAT recordings and their source README
  processed/    CSV recordings grouped by person and activity
  index.csv     Relative file paths and labels: 1 for fall, 0 for non-fall
```

## Get the data

The source is the [HIFD repository](https://github.com/nhoyh/HR_IMU_falldetection_dataset).
From the project root, download it with:

```bash
git clone https://github.com/nhoyh/HR_IMU_falldetection_dataset.git data/hifd/raw
```

Run this only when `data/hifd/raw/` does not already contain the dataset.
Check the source terms before sharing the data with others.

After installing the ML package, create the CSV files and index:

```bash
make prepare-data
```

The raw files are kept. The command writes the processed CSV files and the
index. Paths in `index.csv` are relative to `data/hifd/`.

The local dataset contains 349 recordings from 21 people. See the
[ML guide](../ml/README.md) for the channels and preparation steps used by
the model.
