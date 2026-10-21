# Firmware pentru ceas

Aici va fi codul dispozitivului: colectarea IMU, timestamp-uri, esantionare,
conectivitate, consum energetic si interactiunea locala cu utilizatorul.

Hardware-ul si platforma de compilare nu sunt stabilite. Nu exista inca un
firmware implementat in acest director.

Modelul ML actual asteapta 6 canale IMU la 50 Hz. Folosirea lui pe datele unui
ceas nou necesita verificarea unitatilor, axelor, tratarii gravitatiei si
sincronizarii. Formatul pentru telemetrie si protocolul de comunicare vor fi
definite in `shared/contracts/` dupa alegerea senzorilor si a transportului.
