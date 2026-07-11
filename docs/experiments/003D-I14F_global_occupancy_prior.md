# Expérience 003D-I14F — prior spatial global d'occupation

## Objet

Tester si les candidats attachés aux joueurs appartiennent à des zones
occupées de façon persistante à l'échelle du clip, contrairement aux
candidats balle.

La carte d'occupation est construite sans vérité terrain, à partir de
tous les candidats du clip. La frame du candidat évalué est exclue.

Cette expérience est offline et non causale, car les frames futures du
clip participent à la carte.

## Données

- 3 clips ;
- 450 frames ;
- 416 frames avec balle visible ;
- 10 299 candidats ;
- 359 candidats à moins de 20 px de la balle.

## Caractéristiques

Les caractéristiques mesurent :

- la proportion de frames possédant un candidat proche dans des rayons
  de 16, 32, 64 et 96 px ;
- la densité moyenne de candidats dans des rayons de 32, 64 et 96 px ;
- la distance au candidat d'une autre frame le plus proche.

## Signal descriptif global

Les candidats balle sont généralement situés dans des zones moins
occupées.

| Feature | Médiane balle | Médiane autres | AUC de séparation |
|---|---:|---:|---:|
| Support 16 px | 0,074 | 0,154 | 0,738 |
| Support 32 px | 0,201 | 0,376 | 0,715 |
| Support 64 px | 0,456 | 0,698 | 0,693 |
| Support 96 px | 0,685 | 0,852 | 0,705 |
| Densité 32 px | 0,235 | 0,765 | 0,779 |
| Densité 64 px | 0,631 | 2,523 | 0,763 |
| Densité 96 px | 1,872 | 4,805 | 0,717 |

## Résultats leave-one-clip-out

| Modèle | Hits top-1 | Rappel |
|---|---:|---:|
| Rang actuel | 258 | 62,02 % |
| Intrinsèque | 281 | 67,55 % |
| Occupation seule | 98 | 23,56 % |
| Intrinsèque + occupation | 259 | 62,26 % |

## Résultats par clip

| Clip | Intrinsèque | Intrinsèque + occupation | Écart |
|---|---:|---:|---:|
| `i12a_best_v61_2` | 80 | 64 | -16 |
| `i12a_red_v61_7` | 95 | 87 | -8 |
| `i12a_wide_v61_4` | 106 | 108 | +2 |

Sur `wide`, l'occupation corrige 4 frames et en dégrade 2.

## Faux positifs temporels de `wide`

Les 48 faux positifs temporels de I14C sont beaucoup plus occupés que
les candidats balle du même clip.

| Feature | Faux positifs | Candidats balle |
|---|---:|---:|
| Support 16 px | 0,168 | 0,054 |
| Support 32 px | 0,369 | 0,174 |
| Support 64 px | 0,752 | 0,443 |
| Support 96 px | 0,899 | 0,678 |
| Densité 32 px | 0,876 | 0,215 |
| Densité 64 px | 3,601 | 0,752 |
| Densité 96 px | 6,728 | 1,993 |

Le signal est donc utile à l'intérieur de `wide`, mais ne généralise pas
avec ses valeurs absolues.

## Interprétation

Les distributions d'occupation dépendent fortement :

- du cadrage ;
- de la taille apparente des joueurs ;
- de la résolution ;
- de la quantité de mouvement ;
- de la géométrie propre à chaque clip.

Un modèle leave-one-clip-out utilisant directement les valeurs brutes
confond donc l'identité de scène avec l'identité balle.

## Décision

Le prior d'occupation brut n'est pas intégré au pipeline.

I14F est conservé comme audit positif sur l'existence du signal et
négatif sur sa transférabilité directe.

La prochaine expérience devra convertir les caractéristiques
d'occupation en valeurs relatives au clip :

- percentile ;
- rang ;
- score robuste centré par médiane et écart interquartile.

Cette normalisation devra rester non supervisée et sera évaluée avec le
même protocole leave-one-clip-out.
