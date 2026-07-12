# TTFlux

TTFlux est un pipeline local, reproductible et mesurable d'analyse
vid?o de tennis de table.

## ?tat actuel ? refactor R4 termin?

Le socle logiciel a ?t? r?organis? sans modifier les formats historiques
des runs ni revendiquer une am?lioration de la pr?cision du tracking.

Statut officiel du moteur de balle :

    BALL_TRACKING_BASELINE_FROZEN_EXPLORATORY

La baseline balle est :

- reproductible ;
- instrument?e par des m?triques et des overlays ;
- prot?g?e par des tests de non-r?gression ;
- utilisable comme fondation exp?rimentale ;
- non valid?e comme tracker pr?cis ou final.

La baseline historique ant?rieure au refactor reste conserv?e sur le
commit `4c4dbbf` et sur les branches d'archive associ?es.

## Pipeline canonique actuel

    vid?o locale
      ? extraction reproductible d'un segment
      ? g?n?ration d?terministe de candidats mobiles
      ? scoring interchangeable
      ? pistes temporelles concurrentes
      ? abstention et r?gles de rejet
      ? CSV, JSON, m?triques et overlays

Le moteur canonique reste d?terministe. TTNet et les autres r?seaux
?valu?s dans les documents d'exp?riences restent des baselines externes
ou exp?rimentales ; ils ne sont pas le moteur runtime de TTFlux.

## Refactor R1 ? R4

Le refactor a ?tabli les fronti?res suivantes :

- API publique du tracking sous `ttflux.tracking` ;
- contrats de donn?es et ?tats des runs sous `ttflux.pipeline` ;
- g?n?ration de candidats sous `tracking.candidates` ;
- scoring sous `tracking.candidates.scorers` ;
- pistes temporelles sous `tracking.temporal` ;
- orchestration des runs sous `pipeline.runs` ;
- stockage JSON sous `pipeline.storage` ;
- artefacts sous `pipeline.artifacts` ;
- suppression et maintenance sous `pipeline.maintenance` ;
- extraction des clips sous `pipeline.clips` ;
- rapports d'analyse sous `pipeline.reporting` ;
- catalogue des runs sous `pipeline.catalog`.

Les anciens chemins sous `ttflux.analysis` ne doivent plus recevoir de
nouvelle logique canonique. Ils sont conserv?s uniquement lorsqu'une
compatibilit? historique est n?cessaire.

## R5A ? contrats de sc?ne 2D termin?s

R5A est gel? au commit `60b44d6`.

Le package `ttflux.scene` fournit d?sormais :

- un contrat de point 2D en coordonn?es image ;
- une g?om?trie quadrilat?rale de table ;
- des invariants de bornage, d'ordre et de convexit? ;
- une provenance explicite ;
- une confiance et une incertitude en pixels ;
- un ?tat valide ou invalide explicite ;
- une s?rialisation versionn?e et r?versible.

Ces contrats restent ind?pendants du pipeline, du tracking, de l'interface
web, d'OpenCV et de NumPy.

R5A n'a modifi? ni le moteur de balle ni le contrat historique des runs.

## ?tape active ? R5B

R5B doit uniquement ajouter une persistance optionnelle du contexte de
sc?ne.

Cette ?tape doit :

- d?finir un artefact de sc?ne versionn? ;
- permettre son ?criture et sa lecture atomiques ;
- l'associer facultativement ? un run ;
- pr?server int?gralement les runs historiques sans contexte ;
- ne lancer aucune d?tection ou inf?rence ;
- ne modifier ni `BallTrackingEngine` ni ses m?triques.

La d?tection de table et la r?injection dans le tracking restent diff?r?es.

## Fonctionnalit?s pr?sentes

- structure de projet installable ;
- diagnostic local ;
- catalogue et lecture des vid?os ;
- cr?ation reproductible de runs ;
- cycle `created ? running ? completed / failed` ;
- extraction FFmpeg H.264 d'un segment born? ;
- r?servoir brut de candidats ;
- pistes temporelles exploratoires ;
- abstention et r?gles de rejet explicites ;
- exports CSV, JSON, m?triques et overlays ;
- contrats publics de tracking ;
- tests automatiques.

## Commandes

Cr?er l'environnement :

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
    ??? run.json
    ??? video.json
    ??? source_clip.mp4
    ??? clip.json
    ??? candidates.csv
    ??? candidates_metrics.json
    ??? overlay_candidates.mp4
    ??? tracks_probe.csv
    ??? tracks_metrics.json
    ??? overlay_tracks_probe.mp4
    ??? analysis.json

`candidates.csv` reste la source brute. Les pistes, scorers et exp?riences
ajoutent des hypoth?ses sans supprimer ni r??crire ce r?servoir.

Les documents sous `docs/experiments/` d?crivent l'historique des essais.
Ils ne d?finissent pas ? eux seuls l'?tat courant du moteur runtime.
