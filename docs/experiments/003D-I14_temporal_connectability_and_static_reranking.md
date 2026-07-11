# Expérience 003D-I14 — connectabilité temporelle et reranking statique

## Objet

Déterminer pourquoi le pipeline produit des tracklets majoritairement
attachés aux joueurs, chaussures, mains, têtes, reflets et bords de table,
alors que l'audit I13 montre qu'un candidat proche de la balle existe
souvent dans le réservoir.

L'expérience comporte deux audits :

- I14A : connectabilité temporelle des candidats oracle ;
- I14B : capacité des caractéristiques statiques à reranker les 24
  candidats d'une frame.

## Jeu de vérité terrain

- 3 clips ;
- 450 frames ;
- 416 frames avec balle visible ;
- rayon de validation principal : 20 px.

## I14A — connectabilité temporelle oracle

### Question

Lorsque le bon candidat est fourni au constructeur temporel, les
contraintes cinématiques actuelles permettent-elles de former des pistes ?

### Résultats au cap 24

| Mesure | Valeur |
|---|---:|
| Frames visibles | 416 |
| Frames contenant un candidat oracle | 322 |
| Frames couvertes avec seeding/pruning actuels | 287 |
| Frames couvertes sans restriction de seeding/pruning | 312 |
| Frames oracle restant non connectables | 10 |

Ainsi :

- 89,13 % des frames oracle sont intégrables avec la logique actuelle ;
- 96,89 % sont intégrables lorsque le seeding et le pruning sont libérés.

### Par clip

| Clip | Hits oracle | Couverture actuelle | Couverture libérée | Plus longue piste actuelle | Plus longue piste libérée |
|---|---:|---:|---:|---:|---:|
| `i12a_best_v61_2` | 103 | 93 | 103 | 43 | 45 |
| `i12a_wide_v61_4` | 114 | 97 | 106 | 27 | 30 |
| `i12a_red_v61_7` | 105 | 97 | 103 | 41 | 44 |

### Interprétation

Les contraintes de saut, accélération, prédiction et changement de
direction ne constituent pas le blocage principal.

Quand les bons candidats sont disponibles, le tracker sait généralement
les relier.

Les pertes principales se situent donc avant ou pendant la sélection des
hypothèses :

- candidat balle classé trop bas ;
- candidat absent des six seeds ;
- seed uniquement une frame sur trois ;
- hypothèse balle éliminée par le faisceau au profit d'un distracteur.

Les millions de rejets observés avec le réservoir complet correspondent à
une explosion combinatoire des chemins testés. Ils ne représentent pas
des millions d'erreurs indépendantes.

## I14B — reranking statique leave-one-clip-out

### Protocole

Deux modèles logistiques simples sont entraînés sur deux clips et testés
sur le troisième.

Aucun clip évalué n'est utilisé pour l'apprentissage de son propre modèle.

Les candidats positifs sont ceux situés à 20 px ou moins de la vérité
terrain.

Il existe 359 lignes candidates positives pour 322 frames oracle, car
plusieurs candidats peuvent parfois se trouver dans le rayon de 20 px
d'une même frame.

### Baseline actuelle

| Profondeur | Hits | Rappel |
|---:|---:|---:|
| Top 1 | 258 | 62,02 % |
| Top 3 | 284 | 68,27 % |
| Top 5 | 296 | 71,15 % |
| Top 10 | 312 | 75,00 % |
| Top 24 | 322 | 77,40 % |

### Modèle intrinsèque

Caractéristiques :

- aire logarithmique ;
- luminosité ;
- force du mouvement ;
- taux de remplissage ;
- circularité ;
- dimension maximale ;
- symétrie du rectangle englobant.

| Profondeur | Hits | Rappel | Part du plafond cap 24 |
|---:|---:|---:|---:|
| Top 1 | 281 | 67,55 % | 87,27 % |
| Top 3 | 301 | 72,36 % | 93,48 % |
| Top 5 | 304 | 73,08 % | 94,41 % |
| Top 10 | 313 | 75,24 % | 97,20 % |
| Top 24 | 322 | 77,40 % | 100,00 % |

Le reranking intrinsèque gagne 23 frames au top 1 par rapport à la
baseline.

### Résultat par clip au top 1

| Clip | Baseline | Modèle intrinsèque | Écart |
|---|---:|---:|---:|
| `i12a_best_v61_2` | 61 | 80 | +19 |
| `i12a_red_v61_7` | 86 | 95 | +9 |
| `i12a_wide_v61_4` | 111 | 106 | -5 |

Le gain n'est donc pas uniforme entre domaines.

### Modèle réutilisant le score et le rang actuels

Le modèle ajoutant le score actuel et le rang normalisé obtient seulement :

- 239 hits au top 1 ;
- 57,45 % de rappel ;
- 64 hits sur `i12a_wide_v61_4`.

Cette variante est rejetée.

Le rang courant possède une forte corrélation globale avec la balle, mais
cette relation ne généralise pas correctement entre domaines.

### Séparation individuelle des caractéristiques

Les caractéristiques les plus discriminantes sont :

1. rang actuel, à interpréter comme une mesure descriptive et non comme
   une nouvelle information ;
2. force du mouvement ;
3. score actuel ;
4. circularité ;
5. aire ;
6. dimension ;
7. luminosité ;
8. symétrie ;
9. remplissage.

Les caractéristiques intrinsèques contiennent donc un signal utile, mais
elles ne permettent pas seules une identification fiable de la balle.

## Décision

Le reranking statique intrinsèque est prometteur, mais il n'est pas
intégré au pipeline à ce stade.

Raisons :

- amélioration globale limitée à 5,53 points de rappel ;
- dégradation de 5 frames sur le domaine `wide` ;
- aucune gestion des frames sans balle ;
- aucune prise en compte de la cohérence temporelle ;
- performance encore incompatible avec une trajectoire visuelle fiable
  sur un extrait continu.

La prochaine étape doit combiner :

- caractéristiques intrinsèques ;
- cohérence avec les positions précédentes ;
- coût de transition ;
- concurrence entre hypothèses ;
- abstention lorsqu'aucune piste ne domine clairement.

I14C devra rester un audit hors pipeline avant toute modification du
tracker canonique.
