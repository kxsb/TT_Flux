# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vidéo de tennis de table.

## V0.0

Fonctionnalités présentes :

- structure de projet propre ;
- diagnostic local ;
- catalogue vidéo ;
- lecture des vidéos brutes dans une interface locale ;
- API minimale ;
- tests automatiques.

Le tracking de balle n'est pas encore intégré.

## Commandes

Créer l'environnement :

    py -3.13 -m venv .venv

Installer :

    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Tester :

    .\.venv\Scripts\python.exe -m pytest -q

Diagnostiquer :

    .\.venv\Scripts\python.exe -m ttflux doctor

Lancer l'interface :

    .\.venv\Scripts\python.exe -m ttflux serve


