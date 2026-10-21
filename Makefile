PYTHON ?= .venv/bin/python
CHECKPOINT ?= artifacts/models/inceptiontime/cv/fold_1/checkpoint.pt
RECORDING ?= data/hifd/processed/subject_07/fall/fall1.csv
PORT ?= 5000

.PHONY: setup test prepare-data train smoke predict mlflow
setup:
	$(PYTHON) -m pip install -e './ml[train,dev]'
test:
	$(PYTHON) -m pytest ml/tests -q
prepare-data:
	$(PYTHON) -m safewatch_ml.prepare
train:
	$(PYTHON) -m safewatch_ml.train
smoke:
	$(PYTHON) -m safewatch_ml.train --epochs 1 --fold-limit 1
predict:
	$(PYTHON) -m safewatch_ml.predict --checkpoint "$(CHECKPOINT)" --recording "$(RECORDING)"
mlflow:
	$(PYTHON) -m safewatch_ml.tracking --port $(PORT)
