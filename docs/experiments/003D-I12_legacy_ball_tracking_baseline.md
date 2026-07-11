# Expérience 003D-I12 — baseline historique de tracking de balle

## Objet

Construire un petit jeu de vérité terrain hors domaine et mesurer
objectivement TTNet ainsi que les anciennes sorties TTFlux.

## Jeu de vérité terrain

- 3 clips
- 150 frames par clip
- 450 frames annotées
- 416 frames avec balle visible
- 34 frames sans balle localisable
- audit structurel valide
- aucune frame non annotée

## Résultats TTNet

TTNet raffiné sur les 450 frames :

- rappel à 20 px : 3,37 %
- précision à 20 px : 53,85 %
- F1 à 20 px : 6,33 %
- médiane conditionnelle : 14,94 px

Conclusion : TTNet est parfois précis lorsqu'il répond, mais son rappel
hors domaine est insuffisant.

## Résultats des anciennes sorties TTFlux

Sur les 225 frames communes :

| Variante | Rappel @20 | Précision @20 | F1 @20 |
|---|---:|---:|---:|
| TTNet raffiné | 3,85 % | 72,73 % | 7,31 % |
| TTFlux v63 brut | 1,44 % | 1,33 % | 1,39 % |
| TTFlux v69 calibré | 0,00 % | 0,00 % | 0,00 % |
| TTFlux v72 display | 31,25 % | 35,33 % | 33,16 % |

## Décomposition de v72

| Politique | Rappel @20 | Précision @20 | F1 @20 |
|---|---:|---:|---:|
| strong uniquement | 26,44 % | 46,61 % | 33,74 % |
| strong + weak | 29,33 % | 40,40 % | 33,98 % |
| strong + weak + ghost | 31,25 % | 35,33 % | 33,16 % |

Contribution individuelle :

- strong : 55 hits à 20 px sur 118 prédictions
- weak : 6 hits sur 33 prédictions
- ghost : 4 hits sur 33 interpolations

## Décision

La sortie historique normalisée utilise :

- `strong` comme baseline canonique ;
- `weak` comme candidat secondaire non validé ;
- `ghost` comme interpolation rejetée du signal canonique.

L'ancien pipeline complet n'est pas importé dans le tronc propre.

Les sorties v63 et v69 ne constituent pas des points de reprise utiles.
TTNet n'est pas intégré comme détecteur principal.

## Limites

Le baseline reste insuffisant pour une utilisation réelle :

- rappel canonique à 20 px limité à 26,44 % sur les frames communes ;
- faux positifs sur 41,18 % des frames négatives communes ;
- forte variabilité entre les trois domaines visuels ;
- anciennes sorties disponibles seulement une frame sur deux.

La prochaine amélioration devra partir du réservoir de candidats et du
jeu de vérité terrain, pas du filtre d'affichage v72.
