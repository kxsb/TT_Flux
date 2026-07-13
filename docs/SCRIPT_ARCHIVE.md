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
