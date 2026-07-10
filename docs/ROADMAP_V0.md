# Roadmap TTFlux V0

## État

- [x] V0.0 — Socle propre et bibliothèque vidéo
- [x] V0.1 — Contrat des runs metadata-only
- [x] V0.1.1 — Exécution et cycle d'état metadata-only
- [ ] V0.2 — Baseline balle 2D
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
- métriques vidéo déterministes ;
- persistance d'une erreur structurée en cas d'échec.

## V0.2 — Baseline balle

- candidats par frame ;
- trajectoire sélectionnée ;
- overlay ;
- métriques de couverture et de gaps.

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
