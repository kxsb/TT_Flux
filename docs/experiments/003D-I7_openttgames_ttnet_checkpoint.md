# 003D-I7 — Benchmark OpenTTGames et audit TTNet

## Statut

Checkpoint expérimental après validation automatique du corpus OpenTTGames,
construction d’un benchmark GT figé et évaluation des détecteurs TTFlux et TTNet.

Aucune nouvelle annotation humaine n’est nécessaire pour la prochaine étape.

## Corpus local validé

Source locale canonique :

`C:\Users\micka\Desktop\développement\ping\TTFlux\data_external\sources\openttgames`

Contenu validé :

- 12 vidéos Full HD à 120 fps ;
- `game_1` à `game_5` réservés au développement et à l’entraînement ;
- `test_1` à `test_7` réservés au benchmark final ;
- 52 987 annotations de balle ;
- 52 018 positions de balle visibles ;
- 4 271 événements ;
- masques de segmentation présents ;
- copies locales des annotations identiques.

Convention d’absence OpenTTGames/TTNet :

- une coordonnée `x < 0` ou `y < 0` signifie que la balle est absente.

## Benchmark GT figé

Répertoire local :

`runs\_openttgames_gt_benchmark_003D_I2`

Composition :

- 14 fenêtres de 1 seconde ;
- deux fenêtres pour chaque vidéo `test_1` à `test_7` ;
- 120 images par fenêtre ;
- 1 046 positions GT initiales ;
- 1 030 positions évaluables par le détecteur TTFlux à trois images.

Le split de test ne doit pas être utilisé pour régler ou entraîner un modèle.

## Baseline TTFlux OpenCV

Évaluation automatique 003D-I3 :

- frames GT contenant au moins un candidat : 1,000 ;
- rappel candidats à 5 px : 0,150 ;
- rappel candidats à 10 px : 0,176 ;
- rappel candidats à 20 px : 0,358 ;
- candidat de rang 1 à 20 px : 0,127 ;
- erreur médiane du candidat le plus proche : 187,69 px ;
- rappel couvert par au moins un tracklet à 20 px : 0,095.

Tracklets :

- 336 tracklets sélectionnés ;
- 15 alignés au GT ;
- 13 clean-like ;
- 2 switch-like ;
- pureté médiane des tracklets alignés : 1,000 ;
- meilleure suite liée : 12 points GT.

Conclusion :

Le principal goulot est le détecteur de candidats. L’association temporelle n’est
pas la priorité immédiate.

## Audit d’alignement temporel

Évaluation 003D-I4 des offsets `-4` à `+4` images :

- meilleur offset global : `0` ;
- offset `0` également meilleur pour chacune des sept vidéos de test.

Le faible rappel ne vient pas d’un décalage entre les clips et les annotations.

## Sweep du détecteur heuristique

Répertoire local :

`runs\_openttgames_train_tuning_003D_I5`

Le sweep a été effectué exclusivement sur `game_1` à `game_5`.

Paramètres étudiés :

- gaps temporels : 1, 2 et 3 ;
- seuils de mouvement : 6, 10, 14 et 18 ;
- flou : noyau 1 ou 3 ;
- comparaison top 24 et top 48 candidats.

Meilleure configuration :

- temporal gap : 1 ;
- motion threshold : 18 ;
- blur kernel : 3 ;
- rappel à 20 px, top 24 : 0,404 ;
- rappel à 20 px, top 48 : 0,424 ;
- rappel rang 1 à 20 px : 0,161 ;
- rappel minimal par source : 0,250.

Cette configuration est la baseline actuelle. Les autres configurations sont
moins bonnes.

Conclusion :

La méthode OpenCV à différence symétrique de trois images a atteint son plafond
dans son état actuel. Il ne faut pas poursuivre les balayages de seuils.

## Checkpoints TTNet locaux

Deux checkpoints ont été trouvés :

### TTNet natif

`checkpoints\ttnet_3rd_phase\ttnet_3rd_phase_best.pth`

- SHA-256 :
  `c28cb6e720267c96eed6f2aefbe80c1e13929e6d26171aa6dcfd3d7e58d1df69`
- taille : environ 184,37 Mio ;
- paramètres chargés : 218/218 ;
- epoch enregistré : 3.

### TTNet 30 fps

`checkpoints\ttnet_30fps_3rd_phase\ttnet_30fps_3rd_phase_best.pth`

- SHA-256 :
  `fbd7592639e39d12b3097a93bf58c65924d5b831d51bc44c4f28e65d5d7964f9`
- taille : environ 184,38 Mio ;
- paramètres chargés : 218/218 ;
- epoch enregistré : 1.

Les checkpoints et les vidéos ne sont pas versionnés dans Git.

## Résultat TTNet 003D-I7

### `ttnet_120fps`

Protocole actuel :

- neuf images consécutives ;
- stride temporel 1 ;
- 971 positions GT évaluables.

Sortie globale :

- prédictions valides : 0,993 ;
- rappel à 5 px : 0,000 ;
- rappel à 10 px : 0,001 ;
- rappel à 20 px : 0,020 ;
- erreur médiane : 71,05 px.

Sortie locale raffinée :

- prédictions valides : 0,969 ;
- rappel à 5 px : 0,000 ;
- rappel à 10 px : 0,000 ;
- rappel à 20 px : 0,003 ;
- erreur médiane : 74,17 px.

### `ttnet_30fps`

Protocole actuel :

- neuf images espacées de quatre frames ;
- 676 positions GT évaluables.

Résultat :

- aucune prédiction globale au-dessus du seuil ;
- aucune prédiction raffinée au-dessus du seuil.

## Interprétation prudente

Les résultats TTNet sont anormalement faibles malgré le chargement complet des
poids.

Il ne faut pas conclure immédiatement que les checkpoints sont inutilisables.

Les causes à auditer avant tout réentraînement sont :

1. la reconstruction exacte de la séquence de neuf images ;
2. la frame réellement ciblée par TTNet ;
3. la procédure officielle de décodage global et local ;
4. les paramètres contenus dans les checkpoints ;
5. la convention utilisée par le checkpoint 30 fps ;
6. la différence entre le script expérimental I7 et le pipeline officiel
   d’évaluation TTNet.

## Prochaine étape

Créer 003D-I8 :

- exécuter le pipeline officiel TTNet sur un petit nombre d’échantillons GT ;
- comparer ses tenseurs, coordonnées et erreurs à ceux du script I7 ;
- vérifier la frame cible et le décodage local ;
- ne pas entraîner ;
- ne pas modifier le tracker TTFlux ;
- ne demander aucune nouvelle review humaine.

## Décisions verrouillées

- conserver le détecteur OpenCV comme baseline mesurée ;
- ne plus régler ses seuils pour le moment ;
- ne pas travailler sur la table, les joueurs, le score ou la 3D ;
- ne pas réentraîner TTNet avant l’audit I8 ;
- conserver `test_1` à `test_7` comme benchmark figé.
