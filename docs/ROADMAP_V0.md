# Roadmap TTFlux V0

## État

- [x] V0.0 — Socle propre et bibliothèque vidéo
- [x] V0.1 — Contrat des runs metadata-only
- [x] V0.1.1 — Exécution et cycle d'état metadata-only
- [x] V0.1.2 — Extraction reproductible d'un segment
- [x] V0.1.3 — Réservoir brut de candidats balle
- [x] V0.1.4 — Pistes temporelles exploratoires
- [ ] V0.2 — Baseline balle 2D sélectionnée
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

## V0.2 — Baseline balle sélectionnée

- sélection d'une trajectoire parmi les pistes ;
- métriques de couverture et de gaps ;
- comparaison à une annotation humaine ;
- règles de rejet explicites.

## V0.3 — Review humaine

- navigation frame par frame ;
- correct / wrong / missing / invisible ;
- corrections manuelles ;
- export d'annotations.

## V0.4 — Contexte table

- coins de table ;
- homographie ;
- vue top-down ;
- contraintes géométriques légères.
