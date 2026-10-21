# Componenta ML

Pachetul `safewatch_ml` contine o singura arhitectura activa: InceptionTime
individual in PyTorch, cu 6 module, 16 filtre pe ramura, kerneluri 9/19/39,
conexiuni reziduale si samanta 11. Configuratia se afla in
`configs/inceptiontime.json`.

## Organizare

- `src/safewatch_ml/models.py`: definitia retelei.
- `data.py`: citirea si normalizarea datelor, ferestrele si partitiile pe persoane.
- `prepare.py`: conversia MAT in CSV si generarea indexului.
- `train.py`: antrenarea si urmarirea in MLflow.
- `metrics.py`, `evaluate.py`: praguri, metrici si raportul evaluarii.
- `inference.py`: clasa reutilizabila `FallDetector`, fara importuri MLflow/sklearn.
- `predict.py`: comanda pentru clasificarea unui CSV.
- `tracking.py`: pornirea MLflow cu stocarea in `artifacts/mlflow/`.
- `tests/`: verificari cu semnale sintetice, independente de datasetul local.

Instalare din radacina proiectului:

```bash
.venv/bin/python -m pip install -e './ml[train,dev]'
```

`pyproject.toml` separa dependentele de inferenta de cele optionale `train` si
`dev`. `requirements-lock.txt` pastreaza versiunile mediului experimentului
initial. Pentru reproducerea lui, instaleaza mai intai acel lock, apoi pachetul
cu `pip install --no-deps -e ./ml`.

## Dataset si pregatire

Datasetul local se afla in `data/hifd/raw/`, CSV-urile in
`data/hifd/processed/`, iar indexul este `data/hifd/index.csv`.

```bash
make prepare-data
```

Conversia pastreaza canalele originale. Antrenarea utilizeaza doar
`ax, ay, az, droll, dpitch, dyaw`, in aceasta ordine. Timestamp-urile duplicate
sunt mediate, apoi semnalele sunt interpolate pe o grila de 50 Hz.
Acceleratia folosita in dataset are gravitatia eliminata. Un alt dispozitiv
trebuie sa furnizeze semnale cu aceleasi conventii si unitati.

## Antrenare

```bash
make train
```

Cele 5 folduri sunt separate pe persoane; din fiecare set de dezvoltare se
rezerva alte persoane pentru validare. Normalizarea, selectia epocii si pragul
nu folosesc persoanele de test. O configuratie produce un singur model pentru
fiecare fold. Nu exista selectarea altor arhitecturi sau combinarea modelelor.

Ferestrele au 6 secunde. Pasul este 3 secunde la antrenare si 0,6 secunde la
validare/test. Deoarece etichetele sunt pe inregistrari, pierderea foloseste
maximul logiturilor ferestrelor unei inregistrari (multiple instance learning).
Pragul maximizeaza sensibilitatea pe validare cu specificitate de cel putin
95%; aceasta specificitate nu este garantata pe test.

Rularile noi se salveaza in `artifacts/runs/`. Fiecare checkpoint contine
ponderile, configuratia, normalizarea, pragul si persoanele folosite la evaluare.
Pentru o verificare rapida, `make smoke` limiteaza rularea la 1 epoca si 1 fold;
rezultatul este marcat explicit ca evaluare partiala.

## Folosirea modelului

Din terminal, dupa instalarea pachetului:

```bash
safewatch-predict --checkpoint artifacts/models/inceptiontime/cv/fold_1/checkpoint.pt --recording data/hifd/processed/subject_07/fall/fall1.csv
```

Dintr-o viitoare componenta Python:

```python
from safewatch_ml.inference import FallDetector

detector = FallDetector("artifacts/models/inceptiontime/cv/fold_1/checkpoint.pt")
result = detector.predict_csv("data/hifd/processed/subject_07/fall/fall1.csv")
# Sau: detector.predict_signal(array_samples_by_6_channels, sample_rate=50)
```

Detectorul se incarca o singura data si poate fi reutilizat. Intrarea array are
forma `[numar_esantioane, 6]` si trebuie sa fie esantionata uniform la 50 Hz.
`score` este un scor de clasificare, nu o probabilitate calibrata. Rezultatul
respecta contractul din `shared/contracts/fall-classification.schema.json`.

Toate cele 5 checkpoint-uri InceptionTime din evaluarea originala sunt pastrate
in `artifacts/models/inceptiontime/cv/fold_1` ... `fold_5`. Cel din primul fold
este referinta fixa pentru exemple, nu un model final ales dupa cel mai bun test.
Nu au fost schimbate ponderile in timpul reorganizarii.

## MLflow

```bash
make mlflow
```

Interfata se afla la http://127.0.0.1:5000. Stocarea este
`artifacts/mlflow/mlflow.db`, iar fisierele sunt in
`artifacts/mlflow/artifacts/`. Rularile noi folosesc experimentul
`SafeWatch-InceptionTime`. Experimentele vechi raman vizibile ca istoric.
Pentru alt port: `make mlflow PORT=5050`.

Nu este necesara reantrenarea pentru a folosi checkpoint-urile locale existente.
Pentru alerte in timp real sunt necesare ulterior adnotari temporale, evaluare
pe fluxuri continue si o implementare de preprocesare potrivita dispozitivului.
