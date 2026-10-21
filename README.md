# SafeWatch Companion

Proiectul reuneste ceasul, aplicatia companion si clasificarea caderilor.
Fiecare componenta are propriul director si propriile dependente. In prezent,
componenta implementata este ML; directoarele aplicatiei si firmware-ului sunt
pregatite pentru dezvoltare, fara a impune o platforma sau un framework.

## Structura

```text
SafeWatch_Companion/
  apps/companion/       Aplicatia cu care interactioneaza utilizatorul
  firmware/watch/      Codul care va rula pe ceas
  ml/                  Pachetul Python safewatch_ml: InceptionTime + PyTorch
  shared/contracts/    Formatul comun al rezultatului de clasificare
  docs/                Arhitectura, evaluare ML si surse de cercetare
  reports/ml/          Rezultate compacte care pot fi versionate in Git
  data/                Dataset local, exclus din Git
  artifacts/           Modele, rulari si stocarea MLflow, excluse din Git
  archive/             Comparatia si codul vechi, pastrate doar local
  Makefile             Comenzi uzuale executate din radacina proiectului
```

Separarea permite dezvoltarea aplicatiei fara instalarea bibliotecilor de
antrenare si schimbarea modelului fara rescrierea interfetei utilizatorului.
Locul unde va rula inferenta (ceas, telefon sau serviciu) se va decide dupa
masuratori pe hardware-ul ales.

## Pornire pentru ML

Python 3.11 sau mai nou. Mediul `.venv` exista deja pe calculatorul curent.
Pe o instalare noua, creeaza-l cu `python3 -m venv .venv`.

```bash
make setup          # Instaleaza pachetul ML si dependentele pentru dezvoltare
make test           # Ruleaza testele cu date sintetice
make predict        # Foloseste checkpoint-ul local de referinta
make mlflow         # Deschide serverul local pe http://127.0.0.1:5000
make smoke          # Verifica antrenarea: 1 epoca, 1 fold
make train          # Evalueaza InceptionTime pe toate cele 5 folduri
```

Pe un clone nou, testele functioneaza fara dataset sau ponderi. Antrenarea si
predictia pe date reale necesita restaurarea fisierelor locale descrise in
[data/README.md](data/README.md) si [ml/README.md](ml/README.md).

Singura arhitectura activa este **InceptionTime individual**, cu samanta 11.
Evaluarea anterioara pe 21 de persoane si 349 de inregistrari a obtinut
**97,13% acuratete** si **98,08% sensibilitate**. Acestea sunt rezultate agregate
din validare incrucisata la nivel de inregistrare, nu performanta garantata a
unui checkpoint sau a unei alarme in timp real. Detalii in
[documentatia modelului](docs/ml/model-card.md).

## Git

Repo-ul local este pregatit pe ramura `main`. Pentru prima publicare:

```bash
git status --short
git add .
git commit -m "Organize SafeWatch components and selected ML model"
```

Adauga apoi URL-ul propriului repository cu `git remote add origin URL_REPOSITORY`
si publica folosind `git push -u origin main`.

Sunt versionate codul, configuratiile, testele, contractele, documentatia si
rapoartele compacte. `.gitignore` exclude datasetul, ponderile, bazele MLflow,
arhiva, mediul virtual si secretele. Repo-ul initial nu contine un remote sau
un commit creat automat. Datele si modelele se distribuie separat; Git LFS sau
DVC pot fi introduse ulterior daca apare aceasta nevoie.

## Documentatie

- [Arhitectura si responsabilitatile componentelor](docs/architecture.md)
- [Ghid ML: pregatire, antrenare, inferenta si MLflow](ml/README.md)
- [Modelul selectat si limitele evaluarii](docs/ml/model-card.md)
- [Migrarea vechii structuri](docs/migration-2026-10-21.md)
# SafeWatch
