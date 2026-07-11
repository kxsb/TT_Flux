# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vidéo de tennis de table.

## État actuel — V0.2 en prévalidation, jalon I15A3

Le moteur canonique reste volontairement déterministe :

    vidéo
      → petits composants mobiles sur trois images
      → score heuristique `heuristic_v1`
      → pistes temporelles bornées
      → abstention et règles de rejet
      → exports et évaluation

La baseline V0.2 n'est pas encore déclarée validée. Le travail I12 à I15
a toutefois terminé l'instrumentation nécessaire à sa sélection :

- benchmark humain gelé de 450 images ;
- 416 images avec balle visible et 34 images sans balle visible ;
- réservoir cap 24 de 10 299 candidats ;
- 258 succès top-1 sur 416 images visibles ;
- 322 succès oracle dans le réservoir cap 24 ;
- 340 succès dans le réservoir étendu ;
- audit de connectabilité temporelle et des abstentions ;
- diagnostics visuels des trajectoires et faux positifs ;
- contrat interchangeable `BallCandidateScorer` ;
- conservation exacte de la baseline `heuristic_v1` ;
- manifeste déterministe candidat par candidat :
  322 `ball`, 37 `ignore`, 9 940 `not_ball` ;
- 81 hard negatives issus des faux positifs temporels.

La fin de la prévalidation V0.2 doit encore fixer :

1. la géométrie des crops locaux et contextuels ;
2. leur export reproductible en `t-1 / t / t+1` ;
3. la comparaison leave-one-clip-out des scorers ;
4. l'intégration conservatrice du scorer retenu ;
5. les métriques et le contrôle visuel final du moteur.

## Fonctionnalités présentes

- structure de projet propre ;
- diagnostic local ;
- catalogue et lecture des vidéos brutes ;
- création reproductible de runs d'analyse ;
- cycle d'exécution `created → running → completed / failed` ;
- choix et extraction FFmpeg d'un segment MP4 borné ;
- génération d'un réservoir brut de petits objets mobiles ;
- chaînage temporel de plusieurs pistes concurrentes ;
- abstention et règles de rejet explicites ;
- exports CSV, JSON, métriques et overlays diagnostiques ;
- benchmark humain gelé et audits de non-régression ;
- interface stable de scoring des candidats ;
- tests automatiques.

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

## Contrat canonique d'un run

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

`candidates.csv` reste la source brute. Les pistes et les expériences de
reranking ajoutent des hypothèses sans supprimer ni réécrire ce réservoir.

Les audits I12 à I15 produisent des artefacts reproductibles sous `runs/`.
Ces contenus restent locaux et sont exclus de Git.
