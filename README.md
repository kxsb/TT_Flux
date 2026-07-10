# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vidéo de tennis de table.

## État actuel — V0.1.3

Fonctionnalités présentes :

- structure de projet propre ;
- diagnostic local ;
- catalogue et lecture des vidéos brutes ;
- création reproductible de runs d'analyse ;
- cycle d'exécution `created → running → completed / failed` ;
- choix d'un début et d'une durée depuis l'interface ;
- extraction FFmpeg d'un segment MP4 borné ;
- génération d'un réservoir brut de petits objets mobiles ;
- export `candidates.csv`, métriques et overlay diagnostic ;
- lecture du segment et de l'overlay depuis l'historique ;
- tests automatiques.

Le pipeline courant est `motion_candidates`. Il ne choisit pas encore une
trajectoire de balle. Son rôle est de vérifier que la balle apparaît bien
dans le réservoir de candidats avant d'ajouter du relinking.

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

## Contrat d'un run V0.1.3

    runs/<run_id>/
    ├── run.json
    ├── video.json
    ├── source_clip.mp4
    ├── clip.json
    ├── candidates.csv
    ├── candidates_metrics.json
    ├── overlay_candidates.mp4
    └── analysis.json

Le détecteur utilise une différence temporelle symétrique sur trois
images, puis conserve de petits composants mobiles. Les résultats restent
des candidats bruts, pas une trajectoire validée.

Les contenus de `runs/` restent locaux et sont exclus de Git.
