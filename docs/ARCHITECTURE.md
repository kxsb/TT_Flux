# Architecture TTFlux

## Objectif

TTFlux est un pipeline local, reproductible et mesurable
d'analyse vid?o de tennis de table.

Le d?p?t actif vit sur `main`. Les d?veloppements structurants
sont pr?par?s sur des branches courtes avant int?gration.

## Couches principales

    application
        CLI, interface web, orchestration utilisateur

    pipeline
        cr?ation des runs, cycle d'?tat, composition des moteurs

    tracking
        d?tection de candidats, scoring, pistes temporelles

    scene
        cam?ra, table, filet, joueurs, scoreboard

    evaluation
        benchmarks, m?triques, protocoles de validation

    video
        catalogue, lecture, extraction et m?tadonn?es

## API publique du moteur de balle

Les nouveaux consommateurs doivent importer exclusivement depuis :

    from ttflux.tracking import (
        BallTrackingConfig,
        BallTrackingEngine,
        BallTrackingResult,
    )

Les objets publics sont :

- `BallTrackingConfig` : param?tres complets du moteur ;
- `BallTrackingEngine` : point d'entr?e d'ex?cution ;
- `BallTrackingResult` : r?sultat structur? ;
- `BallTrackingArtifacts` : chemins des artefacts produits ;
- `CandidateConfig` : param?tres du g?n?rateur ;
- `TrackConfig` : param?tres temporels ;
- `BallCandidateScorer` : contrat de scoring.

Les chemins internes sous `ttflux.analysis` ne constituent pas
une API stable.

## Entr?e canonique

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

## Sortie canonique

    BallTrackingResult

Artefacts obligatoires :

- `candidates.csv`
- `candidates_metrics.json`
- `overlay_candidates.mp4`
- `tracks_probe.csv`
- `tracks_metrics.json`
- `overlay_tracks_probe.mp4`

Le moteur ne modifie jamais le clip source.

## R?gles de d?pendance

- `web` et `cli` peuvent d?pendre de `pipeline`.
- `pipeline` peut d?pendre des API publiques des moteurs.
- `tracking` ne d?pend pas de `web`.
- `scene` ne d?pend pas de `web`.
- les exp?riences ne sont pas import?es par le package runtime ;
- les scripts ne d?finissent aucune logique canonique ;
- les param?tres ne doivent pas ?tre cach?s dans les scripts.

## ?tat du moteur de balle

Statut :

    BALL_TRACKING_BASELINE_FROZEN_EXPLORATORY

Le moteur est techniquement stable et reproductible, mais sa
pr?cision n'est pas consid?r?e comme finale.

Il sera r?ouvert apr?s int?gration du contexte de sc?ne.

## Ordre du refactor

1. figer l'API publique et les contrats ;
2. extraire les param?tres et artefacts ;
3. s?parer candidats, scoring et temporalit? ;
4. d?placer l'orchestration des runs vers `pipeline` ;
5. classer les scripts entre validation, maintenance et archives ;
6. ajouter le moteur de sc?ne ;
7. r?injecter le contexte de sc?ne dans le moteur de balle.

Chaque ?tape doit pr?server les tests et les artefacts canoniques.
