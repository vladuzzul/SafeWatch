# Modelul selectat: InceptionTime individual

## Configuratie retinuta

- PyTorch, o singura retea per antrenare, samanta 11.
- 6 module Inception, 16 filtre pe ramura, bottleneck pana la 16 canale.
- Kerneluri temporale 9/19/39, plus ramura MaxPool si proiectie 1x1.
- Conexiuni reziduale dupa fiecare 3 module, agregare globala si un logit.
- 108.133 parametri antrenabili.
- 6 canale IMU: `ax, ay, az, droll, dpitch, dyaw`; 50 Hz, ferestre de 6 secunde.

## Evaluarea originala

21 de persoane, 349 de inregistrari: 104 pozitive si 245 negative. Cinci
partitii externe separate pe persoane si validare interna pe alte persoane.
Epoca si pragul au fost alese exclusiv pe validare. InceptionTime individual
a folosit prima samanta prestabilita, nu membrul ales dupa cel mai bun test.

| Metrica | Rezultat agregat |
|---|---:|
| Acuratete | 339/349 = 97,13% |
| Sensibilitate | 102/104 = 98,08% |
| Specificitate | 237/245 = 96,73% |
| Precizie | 102/110 = 92,73% |
| F1 | 0,9533 |
| Inregistrari pozitive ratate | 2 |
| Inregistrari negative clasificate pozitiv | 8 |

Rapoartele compacte sunt in `reports/ml/inceptiontime-cv-20261021/`.
Cele 5 checkpoint-uri locale sunt in `artifacts/models/inceptiontime/cv/`.
Ele provin din antrenari diferite, pe partitii diferite ale aceleiasi arhitecturi;
nu formeaza un ansamblu in fluxul activ.

Primul checkpoint este folosit ca referinta fixa in exemple. Rezultatul agregat
de 97,13% nu se atribuie acelui checkpoint individual si nu reprezinta evaluarea
unui model final antrenat pe toate persoanele. Pentru productie va fi necesar
un protocol de antrenare finala si un set extern independent.

## Ce prezice

Etichetele disponibile sunt pe inregistrari complete. Modelul este antrenat
prin multiple instance learning: maximul logiturilor ferestrelor este comparat
cu eticheta inregistrarii. Scorurile ferestrelor sunt slab supervizate.

Pragul urmareste sensibilitate mare cu specificitate de minimum 95% pe validare.
Pe date noi, specificitatea poate fi diferita. Scorul sigmoid nu este o
probabilitate calibrata.

## Limite

Datele provin de la adulti tineri si caderi simulate, la incheietura mainii
stangi. Evaluarea nu estimeaza alarmele pe ora sau intarzierea detectarii pe
fluxuri continue. Interpolarea actuala este offline. Integrarea cu un ceas nou
necesita verificarea senzorilor, unitatilor, axelor si tratarii gravitatiei.

Arhitectura este o implementare compacta inspirata de
[InceptionTime](https://github.com/hfawaz/InceptionTime).
Dataset: [HIFD](https://github.com/nhoyh/HR_IMU_falldetection_dataset).
