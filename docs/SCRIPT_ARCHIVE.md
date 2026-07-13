# Archive des scripts expérimentaux

Ce document indexe les scripts retirés de l'arbre actif sans supprimer
leur historique ni leurs artefacts de résultat.

Les sources archivées restent disponibles dans le tag
`ttflux-d1-foundation-20260712` et dans l'historique Git.

## Rendus exploratoires I14

Les trois générateurs de rendus suivants ont été retirés pendant
l'allègement `LIGHT L3C2C` :

- `render_global_occupancy_visual_003D_I14FV.py` ;
- `render_wide_false_positive_gallery_003D_I14D.py` ;
- `render_wide_trajectory_video_003D_I14FV2.py`.

Ils représentaient 2 902 lignes et n'étaient appelés ni par le runtime,
ni par les tests, ni par un autre script.

Les sorties validées restent locales dans :

- `runs/_ball_candidate_global_occupancy_003D_I14F/` ;
- `runs/_ball_false_positive_gallery_003D_I14D/`.

Ce retrait ne modifie aucun protocole I13/I14, aucun artefact suivi par
Git et aucune fonction du moteur TTFlux.

## Diagnostics spatiaux I14E et I14F

Les diagnostics spatiaux I14E et I14F ont été retirés de l'arbre actif
pendant l'allègement `LIGHT L3C3C`.

### I14E — densité locale

- baseline courante top-1 : 258 ;
- modèle intrinsèque top-1 : 281 ;
- meilleure caractéristique spatiale :
  `spatial_nearest_log`, AUC de séparation 0,824089 ;
- ajout des caractéristiques de densité :
  17 corrections, 31 régressions, gain net -14.

### I14F — occupation globale

- baseline courante top-1 : 258 ;
- modèle intrinsèque top-1 : 281 ;
- meilleure caractéristique d'occupation :
  `global_density_32`, AUC de séparation 0,779026 ;
- ajout des caractéristiques d'occupation :
  9 corrections, 31 régressions, gain net -22.

Les caractéristiques spatiales contiennent donc un signal descriptif,
mais leur ajout au modèle intrinsèque dégrade les résultats. Elles ne
sont pas retenues dans le scorer actif.

Les sources restent disponibles dans le tag
`ttflux-d1-foundation-20260712`.

Les rapports et prédictions restent locaux dans :

- `runs/_ball_candidate_density_003D_I14E/` ;
- `runs/_ball_candidate_global_occupancy_003D_I14F/`.

Ce retrait représente 3 246 lignes et ne modifie ni le runtime, ni les
tests, ni les protocoles I13, I14A, I14B ou I14C.

## Protocoles I13B à I14C

Les six protocoles expérimentaux I13B, I13C, I13D, I14A, I14B et I14C
ont été retirés de l'arbre actif pendant l'allègement `LIGHT L3C4D`.

Leur version exacte est conservée dans le tag :

`ttflux-light-i13-i14-protocols-20260713`

### I13B — oracle du réservoir

Sur 416 frames avec balle visible :

- top-1 à 20 px : 258 hits, soit un rappel de 0,620192 ;
- oracle cap 24 : 322 hits, soit un rappel de 0,774038 ;
- 64 frames sont récupérables au-delà du rang 1 ;
- 8 frames visibles ont un réservoir vide.

### I13C — capacité du réservoir

Courbe des hits à 20 px :

| Cap | 1 | 3 | 5 | 10 | 24 | 48 | 96 | 4096 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hits | 258 | 284 | 296 | 312 | 322 | 334 | 340 | 340 |

Le gain devient faible au-delà du cap 48 et plafonne à 340 hits.

### I13D — déterminisme

- divergence de préfixe : 0 frame ;
- divergence entre répétitions : 0 frame ;
- égalité au rang 1 : 0 frame ;
- égalité à la frontière cap 24 : 2 frames.

Le générateur est déterministe pour le protocole mesuré.

### I14A — connectabilité temporelle

| Cap | Mode | Hits | Hits couverts | Plus long chemin |
|---:|---|---:|---:|---:|
| 6 | seeds courants | 299 | 265 | 41 points |
| 6 | oracle libre | 299 | 282 | 42 points |
| 24 | seeds courants | 322 | 287 | 43 points |
| 24 | oracle libre | 322 | 312 | 45 points |
| 4096 | seeds courants | 340 | 309 | 43 points |
| 4096 | oracle libre | 340 | 329 | 45 points |

Le réservoir contient donc des séquences temporellement connectables,
mais le mode à seeds courants ne couvre pas tous les hits disponibles.

### I14B — reranking statique

- baseline courante top-1 : 258 ;
- modèle fondé sur les caractéristiques intrinsèques : 281 ;
- modèle intrinsèque avec réinjection du score courant : 239.

Les caractéristiques intrinsèques améliorent le classement, tandis que
la réinjection du score heuristique courant dégrade le résultat.

### I14C — reranking, abstention et temporalité

| Stratégie | Précision | Rappel | F0.5 | Couverture |
|---|---:|---:|---:|---:|
| classement courant | 0,583710 | 0,620192 | 0,590659 | 0,982222 |
| top-1 intrinsèque | 0,635747 | 0,675481 | 0,643315 | 0,982222 |
| abstention statique | 0,735119 | 0,593750 | 0,701705 | 0,746667 |
| abstention temporelle | 0,732673 | 0,533654 | 0,681818 | 0,673333 |

L'abstention statique offre le meilleur F0.5. L'abstention temporelle
réduit le taux de prédiction sur frames invisibles à 0,676471 et la plus
longue séquence fausse à 16 frames, au prix d'un rappel plus faible.

Aucune stratégie I14B ou I14C n'a été promue comme moteur canonique.

Les rapports et tableaux détaillés restent locaux dans leurs
répertoires `runs/_ball_*_003D_I13*` et `runs/_ball_*_003D_I14*`.

Ce retrait représente 9 085 lignes et ne modifie ni le runtime, ni les
tests, ni le modèle D1 figé.
