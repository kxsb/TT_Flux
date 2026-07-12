# Architecture TTFlux

## Objectif

TTFlux est un pipeline local, reproductible et mesurable d'analyse vid?o
de tennis de table.

L'architecture doit permettre de remplacer un composant sans modifier les
contrats des autres couches ni perdre la reproductibilit? des exp?riences.

## Couches principales

    application
        CLI, interface web et orchestration utilisateur

    pipeline
        runs, ?tats, contrats, stockage et artefacts

    tracking
        candidats, scoring et temporalit?

    scene
        future compr?hension 2D de la table et de son environnement

    video
        catalogue, lecture, extraction et m?tadonn?es

    validation
        tests, benchmarks et documents d'exp?riences

Le package `scene` n'est pas encore impl?ment? au terme de R4.

## API publique du moteur de balle

Les nouveaux consommateurs doivent importer depuis :

    from ttflux.tracking import (
        BallTrackingConfig,
        BallTrackingEngine,
        BallTrackingResult,
    )

Les objets publics comprennent notamment :

- `BallTrackingConfig` : configuration compl?te du moteur ;
- `BallTrackingEngine` : point d'entr?e d'ex?cution ;
- `BallTrackingResult` : r?sultat structur? ;
- `BallTrackingArtifacts` : chemins des artefacts produits ;
- `CandidateConfig` : param?tres du g?n?rateur ;
- `TrackConfig` : param?tres temporels ;
- `BallCandidateScorer` : contrat de scoring.

Les chemins internes sous `ttflux.analysis` ne constituent pas une API
stable. Aucun nouveau composant canonique ne doit y ?tre ajout?.

## Entr?e canonique du tracking

    engine = BallTrackingEngine(config=config)

    result = engine.run(
        clip_path=clip_path,
        output_dir=output_dir,
    )

Entr?es :

- un clip vid?o local ;
- un dossier de sortie ;
- une configuration explicite ;
- un scorer facultatif.

## Sortie canonique du tracking

    BallTrackingResult

Artefacts obligatoires :

- `candidates.csv` ;
- `candidates_metrics.json` ;
- `overlay_candidates.mp4` ;
- `tracks_probe.csv` ;
- `tracks_metrics.json` ;
- `overlay_tracks_probe.mp4`.

Le moteur ne modifie jamais le clip source.

## Modules du pipeline

### `pipeline.runs`

Responsabilit?s conserv?es :

- validation et cr?ation d'un run ;
- orchestration du cycle d'ex?cution ;
- composition du moteur de tracking ;
- wrappers de compatibilit? n?cessaires.

### Modules sp?cialis?s

- `pipeline.contracts` : sch?mas et contrats de donn?es ;
- `pipeline.states` : ?tats et transitions nominales ;
- `pipeline.storage` : lecture et ?criture JSON atomique ;
- `pipeline.artifacts` : r?solution des artefacts ;
- `pipeline.maintenance` : op?rations destructives contr?l?es ;
- `pipeline.clips` : extraction et description des clips ;
- `pipeline.reporting` : provenance et rapport d'analyse ;
- `pipeline.catalog` : lecture et tri de l'historique.

## Modules du tracking

- `tracking.candidates.generator` : g?n?ration d?terministe ;
- `tracking.candidates.scorers` : scoring interchangeable ;
- `tracking.temporal.tracklets` : construction des pistes ;
- API publique expos?e depuis `ttflux.tracking`.

Les modules historiques sous `ttflux.analysis` servent uniquement de
couche de compatibilit? lorsqu'ils sont encore n?cessaires.

## R?gles de d?pendance

- `web` et la CLI peuvent d?pendre de `pipeline` ;
- `pipeline` peut d?pendre des API publiques des moteurs ;
- `tracking` ne d?pend pas de `web` ;
- le futur package `scene` ne d?pendra pas de `web` ;
- `tracking` ne d?pendra pas directement des impl?mentations internes de
  `scene` ;
- les exp?riences ne sont pas import?es par le package runtime ;
- les scripts ne d?finissent aucune logique canonique ;
- les param?tres runtime ne doivent pas ?tre cach?s dans les scripts.

## ?tat du moteur de balle

Statut officiel :

    BALL_TRACKING_BASELINE_FROZEN_EXPLORATORY

Le moteur est techniquement stable et reproductible, mais sa pr?cision
n'est pas consid?r?e comme finale.

Ce statut interdit les formulations suivantes :

- tout marqueur affirmant une validation finale du tracker ;
- ? tracker valid? ? ;
- ? pr?cision finale ? ;
- toute affirmation d'am?lioration non mesur?e.

Le moteur pourra ?tre rouvert apr?s mesure ind?pendante du contexte de
sc?ne.

## Refactor R1 ? R4 ? termin?

Le refactor a :

1. fig? l'API publique et les contrats ;
2. extrait les contrats de sortie ;
3. s?par? candidats, scoring et temporalit? ;
4. d?plac? l'orchestration vers `pipeline` ;
5. extrait ?tats, stockage, artefacts et maintenance ;
6. extrait clips, rapports et catalogue des runs ;
7. conserv? les formats historiques et les alias n?cessaires ;
8. maintenu la suite de tests.

Aucune extraction suppl?mentaire de petite fonction hors de `runs.py`
n'est pr?vue sans besoin fonctionnel concret.

## Point de d?part R5A

R5A cr?e un package `ttflux.scene` minimal, sans mod?le d'inf?rence et
sans modification du tracking existant.

Le premier contrat doit repr?senter une g?om?trie de table 2D observable,
sa provenance, sa validit? et son incertitude.

Les ?tapes ult?rieures pourront ensuite traiter :

1. la persistance d'un contexte de sc?ne ;
2. la d?tection ou l'annotation de la table ;
3. l'agr?gation temporelle pour cam?ra fixe ;
4. l'?valuation ind?pendante ;
5. une int?gration exp?rimentale et r?versible dans le tracking.

La pose humaine, le scoreboard, la 3D et le spin restent hors du p?rim?tre
de R5A.
