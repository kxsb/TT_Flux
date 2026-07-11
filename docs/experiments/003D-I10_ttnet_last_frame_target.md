# 003D-I10 — Correction de la cible temporelle TTNet

Date : 2026-07-11
Statut : `VALIDATED`

## Objet

Vérifier la convention temporelle attendue par les checkpoints TTNet sur le
benchmark OpenTTGames test gelé de TTFlux.

Aucun entraînement n'a été effectué.

## Cause identifiée

L'expérience 003D-I7 construisait une séquence de neuf images centrée sur la
frame portant le GT :

    t-4, t-3, t-2, t-1, t, t+1, t+2, t+3, t+4

Elle comparait ensuite la prédiction à la position de balle de `t`.

Le pipeline d'entraînement TTNet associe toutefois la cible balle à la
dernière image de la séquence.

La séquence correcte pour prédire la balle à la frame `t` est donc :

    t-8, t-7, t-6, t-5, t-4, t-3, t-2, t-1, t

Plus généralement :

    input_frame = (
        target_frame
        - (sequence_length - 1 - sequence_index)
        * temporal_stride
    )

## Protocole

- checkpoint : `ttnet_3rd_phase_best.pth`
- modèle : `ttnet_120fps`
- séquence : 9 images
- stride temporel : 1
- cible : dernière image de la séquence
- résolution source : 1920 × 1080
- résolution TTNet : 320 × 128
- split : OpenTTGames `test_1` à `test_7`
- fenêtres : 14
- positions GT évaluables : 965
- paramètres chargés : 218 / 218
- epoch du checkpoint : 3

## Résultats globaux

| Sortie | Validité | Rappel 5 px | Rappel 10 px | Rappel 20 px | Erreur médiane |
|---|---:|---:|---:|---:|---:|
| Globale | 0.993 | 0.101 | 0.372 | 0.713 | 12.29 px |
| Raffinée | 0.982 | 0.918 | 0.971 | 0.975 | 2.00 px |

## Résultats raffinés par source

| Source | GT | Validité | Rappel 5 px | Rappel 10 px | Rappel 20 px |
|---|---:|---:|---:|---:|---:|
| test_1 | 170 | 0.982 | 0.882 | 0.971 | 0.976 |
| test_2 | 135 | 0.970 | 0.867 | 0.970 | 0.970 |
| test_3 | 134 | 0.978 | 0.955 | 0.978 | 0.978 |
| test_4 | 136 | 1.000 | 1.000 | 1.000 | 1.000 |
| test_5 | 128 | 1.000 | 0.969 | 1.000 | 1.000 |
| test_6 | 149 | 0.987 | 0.913 | 0.973 | 0.980 |
| test_7 | 113 | 0.956 | 0.841 | 0.894 | 0.912 |

## Checkpoint 30 fps

Le checkpoint `ttnet_30fps_3rd_phase_best.pth` produit actuellement zéro
prédiction valide sur les 723 positions évaluables.

Ce résultat est traité séparément. Il ne remet pas en cause la validation du
checkpoint standard 120 fps.

## Décision

`ttnet_120fps` est retenu comme détecteur expérimental pour la prochaine
validation hors domaine OpenTTGames.

Il n'est pas encore intégré au pipeline canonique TTFlux.

La prochaine étape est une validation indépendante sur un petit corpus de
vidéos TTFlux réelles avant toute modification de `candidates.py` ou
`tracks.py`.

## Artefacts locaux

    runs/_ttnet_gt_evaluation_003D_I10/
    ├── evaluate_ttnet_checkpoints.py
    ├── ttnet_gt_predictions.csv
    ├── ttnet_gt_summary.json
    └── execution.txt
