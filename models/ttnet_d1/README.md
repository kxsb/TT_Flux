# TTNet D1 BlurBall Foundation

## Artefact

Fichier : models/ttnet_d1/ttnet_d1_blurball_foundation.pth

Stockage : Git LFS

Taille : 64464651 octets

SHA-256 : 1a165c5775e9edb3c9953c59b26ca7683b72503bf68c70d998f5ec5c97f515d8

## Composition

- étage global TTNet adapté à BlurBall ;
- étage local TTNet historique conservé ;
- étages événements et segmentation inchangés ;
- séquence de 9 frames ;
- entrée réseau 320 x 128 ;
- politique refined_only ;
- seuil global 0.05 ;
- seuil local 0.05.

## Résultat final BlurBall

- précision à 20 px : 0.8201598579040853
- rappel à 20 px : 0.48376113148245153
- F1 à 20 px : 0.60856672158154856
- rappel brut à 20 px : 0.59280600663523664
- erreur médiane valide : 4.5616 px
- faux positifs invisibles : 0.13030746705710103

## Statut

Ce modèle est la fondation de tracking 2D TTFlux.

Il n'est pas considéré comme un modèle universel de compétition.

Le split test BlurBall 22 à 25 est déjà ouvert et ne doit pas servir à
recalibrer cette version.
