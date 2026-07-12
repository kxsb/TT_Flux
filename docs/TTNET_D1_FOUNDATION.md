# TTNet D1 — fondation de tracking 2D TTFlux

## Décision

TTNet D1 est verrouillé comme fondation du pipeline de tracking de balle
TTFlux.

Le modèle est suffisamment robuste sur BlurBall pour servir de base aux
prochaines étapes du projet. Il n'est pas présenté comme un modèle universel
sur toutes les vidéos de compétition.

## Artefact

- emplacement : models/ttnet_d1/ttnet_d1_blurball_foundation.pth
- stockage : Git LFS
- taille : 64464651 octets
- SHA-256 : 1a165c5775e9edb3c9953c59b26ca7683b72503bf68c70d998f5ec5c97f515d8
- tag de fondation : ttflux-d1-foundation-20260712

## Pipeline figé

- adaptation globale D1C ;
- local TTNet historique conservé ;
- politique refined_only ;
- seuil global 0.05 ;
- seuil local 0.05 ;
- aucune recalibration à partir du test D1E ;
- script visuel D1F-R2 versionné dans scripts/visualization/render_ttr_demo_d1f.py.

## Test final BlurBall

| Mesure | TTNet original | TTNet D1 |
|---|---:|---:|
| Précision à 20 px | 0.90222222222222237 | 0.8201598579040853 |
| Rappel à 20 px | 0.017723066177754498 | 0.48376113148245153 |
| F1 à 20 px | 0.034763250278277251 | 0.60856672158154856 |
| Prédictions valides | 225 | 6756 |
| Erreur médiane valide | 2.22825 px | 4.5616 px |

Le test porte sur 80 échanges et
12137 frames par modèle.

## Diagnostic visuel D1F-R2

| Source | Frames | Acceptation |
|---|---:|---:|
| BlurBall reduced_fps.mp4 | 900 | 43.89 % |
| OpenTTGames game_2.mp4 | 3600 | 5.75 % |

Le taux d'acceptation visuel n'est pas une métrique de précision. Il sert à
mesurer qualitativement la capacité du modèle à produire une trajectoire
exploitable.

La forte baisse sur OpenTTGames confirme un transfert de domaine encore
fragile sur les vidéos de compétition brutes.

## Limites connues

- dépendance au domaine visuel BlurBall ;
- sensibilité aux caméras, cadrages et résolutions de compétition ;
- couverture encore faible sur certaines vidéos brutes ;
- faux positifs sur certaines frames sans balle visible ;
- variabilité importante entre matchs.

## Suite du projet

D1 reste immuable.

Les améliorations de généralisation devront être réalisées dans une nouvelle
phase et avec un nouveau protocole de validation. Elles ne modifieront pas
rétroactivement les résultats D1.
