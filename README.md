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

## ?tape active ? R5A

La prochaine ?tape unique est la cr?ation d'un socle de compr?hension de
sc?ne 2D.

R5A doit uniquement introduire :

- un package `ttflux.scene` minimal ;
- des contrats explicites pour la g?om?trie 2D de la table ;
- la provenance et l'incertitude des observations de sc?ne ;
- des tests unitaires de s?rialisation et d'invariants g?om?triques.

R5A ne doit pas encore :

- modifier le moteur de balle ;
- entra?ner un nouveau r?seau ;
- int?grer la pose humaine ;
- reconstruire la sc?ne en 3D ;
- estimer le spin ;
- ajouter une nouvelle logique dans l'interface web.

Le contexte de sc?ne sera r?inject? dans le tracking seulement apr?s avoir
?t? mesur? ind?pendamment.

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
