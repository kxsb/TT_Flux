# Roadmap TTFlux V0

## Statut global

Statut officiel du moteur de balle :

    BALL_TRACKING_BASELINE_FROZEN_EXPLORATORY

La baseline balle est fig?e comme fondation exploratoire. Elle n'est pas
d?clar?e valid?e en pr?cision.

L'?tape active est R5B : persistance optionnelle du contexte de sc?ne.

## Jalons termin?s

- [x] V0.0 ? Socle propre et biblioth?que vid?o
- [x] V0.1 ? Contrat des runs metadata-only
- [x] V0.1.1 ? Ex?cution et cycle d'?tat
- [x] V0.1.2 ? Extraction reproductible d'un segment
- [x] V0.1.3 ? R?servoir brut de candidats balle
- [x] V0.1.4 ? Pistes temporelles exploratoires
- [x] I12 ? I16 ? Instrumentation et gel exploratoire de la baseline
- [x] R1 ? R3 ? API et modules internes du tracking
- [x] R4A ? R4G ? Modularisation du pipeline et des runs
- [x] R5A ? Contrats minimaux de sc?ne 2D

## Baseline balle 2D ? ?tat fig?

Le travail exp?rimental a produit :

- un benchmark humain gel? de 450 images ;
- 416 images avec balle visible et 34 sans balle visible ;
- un r?servoir cap 24 de 10 299 candidats ;
- 258 succ?s top-1 sur 416 images visibles ;
- 322 succ?s oracle dans le r?servoir cap 24 ;
- 340 succ?s dans le r?servoir ?tendu ;
- des audits de d?terminisme et de connectabilit? ;
- des diagnostics sur les faux positifs et les abstentions ;
- un contrat interchangeable `BallCandidateScorer` ;
- une conservation exacte de `heuristic_v1` ;
- un manifeste candidat d?terministe ;
- des labels `ball / not_ball / ignore` ;
- 81 hard negatives temporels ;
- des overlays et rapports reproductibles.

Ces r?sultats mesurent le potentiel et les limites du r?servoir. Ils ne
constituent pas une validation finale du tracker.

Les exp?riences de reranking I14 et les essais TTNet restent document?s,
mais aucun de ces mod?les n'est d?clar? moteur canonique valid?.

## D?cision de gel

La piste consistant ? s?lectionner imm?diatement un nouveau scorer est
suspendue.

Motifs :

- la sc?ne produit encore de nombreux objets concurrents ;
- les faux positifs li?s aux joueurs et ? l'environnement restent
  structurants ;
- une optimisation locale du scorer risque d'apprendre l'identit? du
  corpus plut?t que celle de la balle ;
- le contexte de table et de sc?ne doit ?tre mesur? s?par?ment avant
  r?int?gration.

Le r?servoir brut, les m?triques et la baseline d?terministe restent
inchang?s afin de servir de r?f?rence.

## Refactor R1 ? R4 ? termin?

### R1 et R2

- [x] API publique du tracking ;
- [x] contrats des r?sultats et artefacts ;
- [x] d?pendances historiques prot?g?es.

### R3

- [x] scorers sous `tracking.candidates.scorers` ;
- [x] g?n?rateur sous `tracking.candidates.generator` ;
- [x] tracklets sous `tracking.temporal.tracklets` ;
- [x] imports publics paresseux.

### R4

- [x] orchestration d?plac?e sous `pipeline.runs` ;
- [x] contrats et ?tats extraits ;
- [x] stockage JSON extrait ;
- [x] artefacts et maintenance extraits ;
- [x] gestion des clips extraite ;
- [x] reporting extrait ;
- [x] catalogue des runs extrait ;
- [x] formats des runs pr?serv?s ;
- [x] suite de tests pr?serv?e.

## R5 ? Compr?hension de sc?ne 2D

### R5A ? Contrats de sc?ne ? termin?

Gel? au commit `60b44d6`.

- [x] cr?er `src/ttflux/scene/` ;
- [x] d?finir un contrat de g?om?trie de table 2D ;
- [x] repr?senter provenance, validit? et incertitude ;
- [x] d?finir une s?rialisation stable ;
- [x] tester les invariants g?om?triques ;
- [x] ne modifier ni `BallTrackingEngine` ni les artefacts actuels ;
- [x] rester importable sans d?pendance lourde ;
- [x] rester ind?pendant de l'interface web ;
- [x] n'ex?cuter aucune inf?rence ;
- [x] pr?server les tests historiques.

### R5B ? Persistance du contexte ? actif

Objectif actuel et unique :

- [ ] d?finir un artefact JSON de sc?ne versionn? ;
- [ ] ?crire et lire cet artefact atomiquement ;
- [ ] associer le contexte de mani?re optionnelle ? un run ;
- [ ] pr?server les runs historiques d?pourvus de contexte ;
- [ ] ne lancer aucune d?tection ou inf?rence ;
- [ ] ne modifier ni le tracking ni ses m?triques.

### R5C ? Observation de la table

Apr?s validation de R5B seulement :

- importer, annoter ou d?tecter les points de table ;
- agr?ger les observations d'une cam?ra fixe ;
- mesurer erreurs et taux d'abstention ind?pendamment du tracking balle.

### R5D ? Int?gration exp?rimentale

Apr?s mesure ind?pendante de la sc?ne :

- injecter le contexte par une interface optionnelle ;
- comparer contre la baseline gel?e ;
- conserver un mode strictement sans contexte ;
- accepter uniquement une am?lioration mesur?e et reproductible.

## ?tapes diff?r?es

Ces chantiers ne doivent pas ?tre ouverts pendant R5B :

- interface compl?te de review humaine ;
- analyse du scoreboard ;
- pose et biom?canique des joueurs ;
- reconstruction 3D ;
- estimation du spin ;
- moteur physique ;
- entra?nement d'un nouveau scorer balle.

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

R5A ne modifie pas ce contrat.

## Documents historiques

Les fichiers sous `docs/experiments/` sont des rapports immuables des
exp?riences r?alis?es. Ils peuvent contenir des conclusions interm?diaires
qui ne repr?sentent plus la direction active du projet.
