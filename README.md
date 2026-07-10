# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vidéo de tennis de table.

## État actuel — V0.1.1

Fonctionnalités présentes :

- structure de projet propre ;
- diagnostic local ;
- catalogue et lecture des vidéos brutes ;
- création reproductible de runs d'analyse ;
- cycle d'exécution `created → running → completed / failed` ;
- contrats `run.json`, `video.json` et `analysis.json` ;
- historique local des analyses ;
- API minimale ;
- tests automatiques.

Le tracking de balle n'est pas encore intégré. Le pipeline courant est
`metadata_only` : il vérifie l'orchestration et produit des métriques
déterministes à partir des métadonnées vidéo.

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

## Contrat d'un run V0.1.1

    runs/<run_id>/
    ├── run.json
    ├── video.json
    └── analysis.json

`analysis.json` contient actuellement :

- nombre d'images ;
- durée et cadence ;
- résolution ;
- intervalle temporel estimé entre deux images.

Les contenus de `runs/` restent locaux et sont exclus de Git.
