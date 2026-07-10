# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vidéo de tennis de table.

## État actuel — V0.1.2

Fonctionnalités présentes :

- structure de projet propre ;
- diagnostic local ;
- catalogue et lecture des vidéos brutes ;
- création reproductible de runs d'analyse ;
- cycle d'exécution `created → running → completed / failed` ;
- choix d'un début et d'une durée depuis l'interface ;
- extraction FFmpeg d'un segment MP4 borné ;
- contrats `run.json`, `video.json`, `clip.json` et `analysis.json` ;
- lecture du segment extrait depuis l'historique ;
- tests automatiques.

Le tracking de balle n'est pas encore intégré. Le pipeline courant est
`clip_extract`. Il prépare un segment court et reproductible qui servira
d'entrée à la première baseline balle 2D.

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

## Contrat d'un run V0.1.2

    runs/<run_id>/
    ├── run.json
    ├── video.json
    ├── source_clip.mp4
    ├── clip.json
    └── analysis.json

Le segment est réencodé en H.264, sans audio, avec une durée comprise
entre 1 et 60 secondes.

Les contenus de `runs/` restent locaux et sont exclus de Git.
