# Expérience 003D-I14C — reranking temporel avec abstention

## Objet

Évaluer si la combinaison suivante permet d'identifier la balle de façon
plus fiable que le classement statique actuel :

- score intrinsèque appris en leave-one-clip-out ;
- contraintes de saut, prédiction, accélération et changement de direction ;
- cohérence d'apparence ;
- acquisition confirmée sur plusieurs frames ;
- abstention lorsque la confiance est insuffisante.

L'expérience reste hors pipeline canonique.

## Données

- 3 clips ;
- 450 frames ;
- 416 frames avec balle visible ;
- 34 frames sans balle visible ;
- 10 299 lignes candidates ;
- rayon de validation : 20 pixels.

## Protocole

Pour chaque clip d'évaluation :

1. le modèle intrinsèque est entraîné sur les deux autres clips ;
2. les seuils d'abstention et de continuité sont sélectionnés uniquement
   à partir des clips d'entraînement ;
3. le clip tenu à l'écart est évalué sans réajustement.

La métrique de sélection est le F0.5, puis la précision, puis le rappel.

## Résultats globaux

| Stratégie | Prédictions | Précision | Rappel | F0.5 | Plus longue erreur |
|---|---:|---:|---:|---:|---:|
| Rang 1 actuel | 442 | 58,37 % | 62,02 % | 59,07 % | 36 |
| Intrinsèque top 1 | 442 | 63,57 % | 67,55 % | 64,33 % | 36 |
| Statique avec abstention | 336 | 73,51 % | 59,38 % | 70,17 % | 36 |
| Temporel avec abstention | 303 | 73,27 % | 53,37 % | 68,18 % | 16 |

## Effet du modèle temporel

Le modèle temporel ne dépasse pas le reranking statique sur le F0.5 :

- statique avec abstention : 70,17 % ;
- temporel avec abstention : 68,18 %.

Il apporte toutefois des améliorations structurelles :

- médiane des séquences correctes : 7,5 frames contre 3,5 ;
- plus longue erreur continue : 16 frames contre 36 ;
- plus longue séquence correcte : 38 frames ;
- prédictions sur frames invisibles : environ 23 sur 34, contre 31 sur 34
  pour l'abstention statique et 34 sur 34 sans abstention.

Le temporel réduit donc les longues dérives, mais ne sait pas encore
détecter correctement l'absence de balle.

## Résultat temporel par clip

| Clip | Prédictions | TP | FP | Précision | Rappel | Plus longue séquence correcte | Plus longue erreur |
|---|---:|---:|---:|---:|---:|---:|---:|
| `i12a_best_v61_2` | 59 | 59 | 0 | 100,00 % | 40,14 % | 24 | 0 |
| `i12a_red_v61_7` | 134 | 101 | 33 | 75,37 % | 84,87 % | 38 | 16 |
| `i12a_wide_v61_4` | 110 | 62 | 48 | 56,36 % | 41,33 % | 23 | 13 |

## Instabilité des seuils

Les paramètres choisis changent fortement selon le clip tenu à l'écart.

### Abstention statique

- `best` : seuil 0,8 ;
- `red` : seuil 0,6 ;
- `wide` : seuil 0,4.

### Acquisition temporelle

- `best` : seuil 0,65, confirmation sur 3 frames ;
- `red` : seuil 0,65, confirmation sur 2 frames ;
- `wide` : seuil 0,45, confirmation sur 2 frames.

Cette variation traduit un décalage de distribution entre clips. Les
probabilités intrinsèques ne sont pas suffisamment calibrées pour être
utilisées avec un seuil global.

## Décision

I14C est conservé comme audit, mais aucune variante n'est intégrée au
pipeline.

Le reranking statique avec abstention obtient le meilleur compromis
global, tandis que la logique temporelle améliore la longueur des
séquences et limite les longues dérives.

Les limitations restantes sont :

- forte dépendance au domaine vidéo ;
- mauvaise détection des frames sans balle ;
- faible précision sur le clip `wide` ;
- seuils non transférables ;
- erreurs visuelles probablement liées à des distracteurs persistants.

La prochaine étape doit être un diagnostic ciblé des faux positifs du
clip `wide`, afin de déterminer s'ils correspondent principalement aux
joueurs, aux lignes de table, au filet, aux reflets ou à d'autres objets
mobiles.

Aucun nouveau modèle ne doit être intégré avant cette analyse.
