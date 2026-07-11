# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vidéo de tennis de table.

## État actuel — V0.1.4

Fonctionnalités présentes :

- structure de projet propre ;
- diagnostic local ;
- catalogue et lecture des vidéos brutes ;
- création reproductible de runs d'analyse ;
- cycle d'exécution `created → running → completed / failed` ;
- choix et extraction FFmpeg d'un segment MP4 borné ;
- génération d'un réservoir brut de petits objets mobiles ;
- chaînage temporel exploratoire de plusieurs pistes concurrentes ;
- exports CSV, métriques et overlays diagnostiques ;
- lecture des artefacts vidéo depuis l'historique ;
- tests automatiques.

Le pipeline courant est `motion_tracks_probe`. Il conserve les candidats
bruts, puis construit jusqu'à dix pistes concurrentes avec une recherche
temporelle légère, une prédiction de vitesse et une tolérance de trois
images manquantes.

Ces pistes ne constituent pas encore une trajectoire de balle validée.
Elles servent à mesurer si la continuité temporelle sépare la balle des
mouvements des joueurs.

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

## Contrat d'un run V0.1.4

    runs/<run_id>/
    ├── run.json
    ├── video.json
    ├── source_clip.mp4
    ├── clip.json
    ├── candidates.csv
    ├── candidates_metrics.json
    ├── overlay_candidates.mp4
    ├── tracks_probe.csv
    ├── tracks_metrics.json
    ├── overlay_tracks_probe.mp4
    └── analysis.json

`candidates.csv` reste la source brute. `tracks_probe.csv` ajoute des
hypothèses temporelles sans supprimer ni réécrire les candidats.

Les contenus de `runs/` restent locaux et sont exclus de Git.
