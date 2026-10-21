# Date locale

Datele sunt excluse din Git. Structura locala:

```text
hifd/
  raw/          Datasetul original MAT, cu README-ul si provenienta sa
  processed/    CSV-uri organizate pe subject_NN/fall sau non-fall
  index.csv     Cai relative la acest director si etichetele 0/1
```

Sursa: [HIFD, repository-ul autorilor](https://github.com/nhoyh/HR_IMU_falldetection_dataset).
Cloneaza sursa in `data/hifd/raw/`, apoi executa `make prepare-data` din radacina
proiectului pentru a genera CSV-urile si indexul. Verifica termenii sursei inainte
de redistribuire. Clone-ul local al datasetului isi pastreaza istoricul Git
propriu; directorul sau este ignorat de repo-ul SafeWatch.
