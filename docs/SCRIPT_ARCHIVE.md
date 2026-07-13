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
