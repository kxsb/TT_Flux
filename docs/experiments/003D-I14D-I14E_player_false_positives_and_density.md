# Expériences 003D-I14D–I14E — faux positifs joueur et densité locale

## I14D — revue visuelle des faux positifs sur `wide`

### Données

Le modèle temporel avec abstention de I14C produit sur
`i12a_wide_v61_4` :

- 110 prédictions ;
- 62 prédictions correctes ;
- 48 faux positifs ;
- 18 séquences erronées ;
- plus longue séquence erronée : 13 frames.

Les 48 faux positifs ont lieu sur des frames où la balle est visible.

### Observation visuelle

La galerie montre que les erreurs sont principalement attachées aux
joueurs :

- tête ;
- bras, main ou raquette ;
- torse ;
- jambes ou chaussures.

Aucune famille dominante de faux positifs liée au filet, aux lignes de
table ou aux reflets n'est visible dans cette séquence.

Les deux dérives principales sont :

- `R007` : 7 frames sur le joueur de droite ;
- `R014` : 13 frames sur le joueur de gauche.

La continuité temporelle stabilise donc parfois une mauvaise identité
visuelle déjà plausible.

## I14E — densité locale des candidats

### Hypothèse

Les candidats provenant des joueurs pourraient appartenir à des nuages
locaux denses, tandis que la balle pourrait être plus isolée.

### Protocole

Évaluation leave-one-clip-out sur les 416 frames GT visibles.

Trois modèles sont comparés :

- caractéristiques intrinsèques ;
- densité locale seule ;
- caractéristiques intrinsèques et densité locale.

### Signal descriptif

Les candidats balle sont généralement plus isolés :

| Feature | Médiane balle | Médiane autres | AUC de séparation |
|---|---:|---:|---:|
| Voisins à 20 px | 0 | 1 | 0,759 |
| Voisins à 40 px | 0 | 3 | 0,822 |
| Voisins à 80 px | 0 | 6 | 0,798 |
| Distance au voisin le plus proche | 4,395 log | 2,664 log | 0,824 |
| Nombre temporel à 48 px | 1,0 | 3,5 | 0,797 |

Les mesures simples de support temporel sont beaucoup moins utiles, avec
des AUC proches de 0,5.

### Résultats top-1

| Modèle | Hits | Rappel |
|---|---:|---:|
| Classement actuel | 258 | 62,02 % |
| Intrinsèque | 281 | 67,55 % |
| Densité seule | 173 | 41,59 % |
| Intrinsèque + densité | 267 | 64,18 % |

### Résultats par clip

| Clip | Intrinsèque | Intrinsèque + densité | Écart |
|---|---:|---:|---:|
| `i12a_best_v61_2` | 80 | 64 | -16 |
| `i12a_red_v61_7` | 95 | 99 | +4 |
| `i12a_wide_v61_4` | 106 | 104 | -2 |

Sur `wide`, la densité corrige 5 frames mais en dégrade 7.

### Conclusion

L'isolation spatiale contient un signal descriptif réel, mais ce signal
n'est pas suffisamment stable entre domaines pour améliorer le
reranking.

La densité locale ne doit pas être intégrée au pipeline.

## Décision

Les scripts I14D et I14E sont conservés comme audits.

La prochaine étape doit tester une information de scène plus globale :
une carte non supervisée des zones où des candidats persistent à
l'échelle du clip.

Cette carte doit rester un prior doux. Elle ne devra pas supprimer
automatiquement les candidats proches d'un joueur, car la balle peut
passer devant le corps, la main ou la raquette.

Un détecteur explicite de personne ne sera envisagé que si ce prior
spatial global reste insuffisant.
