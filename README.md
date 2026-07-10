# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vidéo de tennis de table.

## État actuel — V0.1

Fonctionnalités présentes :

- structure de projet propre ;
- diagnostic local ;
- catalogue et lecture des vidéos brutes ;
- création reproductible de runs d'analyse ;
- contrats `run.json` et `video.json` ;
- historique local des analyses ;
- API minimale ;
- tests automatiques.

Le tracking de balle n'est pas encore intégré. Le pipeline courant est
`metadata_only`.

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

Adresse : http://127.0.0.1:8787

## Contrat d'un run V0.1

    runs/<run_id>/
    ├── run.json
    └── video.json

Les contenus de `runs/` restent locaux et sont exclus de Git.
