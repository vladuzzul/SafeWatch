# Contracte intre componente

Contractul `fall-classification.schema.json` descrie rezultatul curent al
clasificarii unei inregistrari: scor, prag, decizie 0/1 si versiunea formatului.
Este independent de framework-ul interfetei.

`prediction=1` indica faptul ca modelul considera ca inregistrarea contine o
cadere. Acest rezultat nu defineste inca un eveniment de urgenta sau o comanda
de notificare. Fluxul de confirmare, identificatorii dispozitivelor si protocolul
de transport vor fi definite odata cu aplicatia si ceasul.
