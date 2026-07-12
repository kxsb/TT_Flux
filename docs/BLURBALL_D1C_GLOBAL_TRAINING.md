# BlurBall D1C — entraînement global TTNet

## Statut

D1C est validé.

Le checkpoint adapte uniquement l'étage global de détection de balle de TTNet.
Les branches locale, événement et segmentation sont restées inchangées.

## Corpus

| Ensemble | Matchs | Échanges | Fenêtres évaluables |
|---|---:|---:|---:|
| Train | 00–19 | 353 | 43 040 |
| Validation | 20–21 | 30 | 7 714 |
| Test gelé | 22–25 | 80 | non utilisé |

L'entraînement a traité 49 804 fenêtres après suréchantillonnage contrôlé
des frames invisibles.

## Configuration

| Paramètre | Valeur |
|---|---:|
| Epochs | 1 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Fenêtre temporelle | 9 frames |
| Cible temporelle | dernière frame |
| Seuil de prédiction | 0.05 |
| BatchNorm | statistiques gelées |
| Dropout | désactivé |
| Étapes totales | 3 267 |
| Meilleure étape | 3 000 |
| Temps d'entraînement | 498.581 s |

## Résultats validation

| Mesure | Checkpoint initial | D1C étape 3000 |
|---|---:|---:|
| Loss | 0.12208388657988833 | 0.06611797756785641 |
| Rappel brut @20 px | 0.05993558325164543 | 0.5915137935863325 |
| Rappel @20 px | 0.03164822853942025 | 0.5642066937403725 |
| Précision @20 px | 0.5854922279792746 | 0.7931102362204724 |
| F1 @20 px | 0.06005048492095124 | 0.6593568447753867 |
| Erreur médiane | 350.221762637618 px | 13.960121775973182 px |
| Erreur P90 | 670.042231579473 px | 187.21738494327917 px |
| Faux positifs sur frames invisibles | 0.04013961605584642 | 0.2181500872600349 |

## Checkpoint local

Chemin :

``text
runs/dataset_training/d1c_blurball_global_full/d1c_ttnet_global_best.pth
Taille : 61.478 MiB
SHA-256 : 1a165c5775e9edb3c9953c59b26ca7683b72503bf68c70d998f5ec5c97f515d8

Le checkpoint n'est pas ajouté à Git.

Interprétation

L'adaptation du seul étage global produit une généralisation nette entre les
matchs d'entraînement et les matchs de validation.

Le meilleur état est atteint avant la fin de l'epoch, à l'étape 3000.
La dégradation observée à l'étape 3267 confirme l'utilité de la sélection
du meilleur checkpoint sur la validation.

La précision atteint environ 79 %, mais le taux de réponses sur les frames
invisibles atteint environ 21,8 %. La prochaine phase doit donc entraîner
l'étage local avec l'étage global D1C gelé, puis calibrer la décision finale
exclusivement sur la validation.

Limite de la validation

Les matchs de validation 20 et 21 sont à 60 fps. D1C ne constitue donc pas
encore une validation équilibrée entre les cadences 24/25, 30 et 60 fps.

Étape suivante

D1D :

charger le checkpoint D1C ;
geler l'étage global ;
entraîner ball_local_stage ;
utiliser des crops compatibles avec l'inférence ;
conserver événements et segmentation inchangés ;
sélectionner le meilleur état sur les matchs 20 et 21 ;

ne pas consulter les matchs test 22 à 25.
