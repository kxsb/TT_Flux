# Roadmap TTFlux V0

## État

- [x] V0.0 — Socle propre et bibliothèque vidéo
- [x] V0.1 — Contrat des runs metadata-only
- [x] V0.1.1 — Exécution et cycle d'état metadata-only
- [x] V0.1.2 — Extraction reproductible d'un segment
- [x] V0.1.3 — Réservoir brut de candidats balle
- [x] V0.1.4 — Pistes temporelles exploratoires
- [ ] V0.2 — Baseline balle 2D sélectionnée — prévalidation I15A3
- [ ] V0.3 — Review humaine
- [ ] V0.4 — Contexte table

## V0.0 — Socle propre

- package Python installable ;
- diagnostic de l'environnement ;
- catalogue des vidéos ;
- interface de lecture locale.

## V0.1 — Contrat des analyses

- création d'un run ;
- configuration sauvegardée ;
- métadonnées vidéo ;
- état du traitement ;
- historique des analyses.

## V0.1.1 — Exécution metadata-only

- cycle `created → running → completed / failed` ;
- écriture atomique des états ;
- production de `analysis.json` ;
- persistance d'une erreur structurée en cas d'échec.

## V0.1.2 — Segment d'analyse

- début et durée choisis depuis le lecteur ;
- validation d'une plage de 1 à 60 secondes ;
- extraction FFmpeg H.264 ;
- production de `source_clip.mp4` et `clip.json` ;
- lecture du segment depuis l'historique.

## V0.1.3 — Réservoir de candidats

- différence temporelle symétrique sur trois images ;
- filtrage de petits composants mobiles ;
- classement local par score ;
- `candidates.csv` ;
- `candidates_metrics.json` ;
- `overlay_candidates.mp4`.

## V0.1.4 — Pistes temporelles exploratoires

- recherche en faisceau bornée ;
- connexion entre images voisines ;
- prédiction par vitesse récente ;
- tolérance de trois images manquantes ;
- conservation de plusieurs pistes concurrentes ;
- `tracks_probe.csv` ;
- `tracks_metrics.json` ;
- `overlay_tracks_probe.mp4`.

## V0.2 — Baseline balle 2D sélectionnée

### Prévalidation terminée

- [x] benchmark humain gelé de 450 images ;
- [x] comparaison top-1, cap 24 et réservoir étendu ;
- [x] audit de déterminisme du générateur de candidats ;
- [x] audit de connectabilité temporelle ;
- [x] audit des faux positifs et des abstentions ;
- [x] diagnostics visuels des trajectoires ;
- [x] contrat interchangeable `BallCandidateScorer` ;
- [x] non-régression exacte de `heuristic_v1` ;
- [x] manifeste candidat déterministe ;
- [x] labels `ball / not_ball / ignore` ;
- [x] identification de 81 hard negatives temporels.

### Travail restant avant validation V0.2

- [ ] fixer la géométrie locale et contextuelle des crops ;
- [ ] exporter les crops `t-1 / t / t+1` ;
- [ ] contrôler visuellement les crops et les labels ;
- [ ] comparer les scorers en leave-one-clip-out ;
- [ ] sélectionner le scorer final ;
- [ ] intégrer le gagnant sans dégrader la baseline gelée ;
- [ ] geler les métriques de couverture, gaps et erreurs ;
- [ ] produire le MP4 comparatif de validation ;
- [ ] déclarer la baseline V0.2 validée.

### Critères de sortie

- une trajectoire canonique est sélectionnée ;
- les métriques sont comparées au benchmark humain ;
- les règles de rejet et d'abstention sont explicites ;
- les résultats sont reproductibles ;
- le contrôle visuel final est validé.

## V0.3 — Review humaine

- navigation frame par frame ;
- correct / wrong / missing / invisible ;
- corrections manuelles ;
- export d'annotations.

Le benchmark gelé I12 prépare cette étape, mais ne remplace pas encore
l'interface complète de review et de correction.

## V0.4 — Contexte table

- coins de table ;
- homographie ;
- vue top-down ;
- contraintes géométriques légères.

Le contexte table ne doit être réintégré au moteur canonique qu'après la
validation de la baseline balle V0.2.
