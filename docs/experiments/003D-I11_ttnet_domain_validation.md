# 003D-I11 — Validation hors domaine du checkpoint TTNet

Date : 2026-07-11
Statut : `COMPLETED`

## Objet

Évaluer le checkpoint standard `ttnet_120fps` sur des vidéos de tennis de
table différentes du domaine OpenTTGames utilisé pour I10.

Aucun entraînement et aucune modification du moteur canonique TTFlux n'ont
été effectués.

## Corpus I11B

Six sources indépendantes ont été retenues parmi neuf sources disponibles.

Caractéristiques communes :

- fréquence : 50 fps ;
- résolution : 1280 × 720 ;
- durée par clip : 38 secondes ;
- cadrages, salles, éclairages et arrière-plans variés ;
- absence de GT de balle exploitable pour cette première sonde.

Sources retenues :

| Source |
|---|
| `v61_1_M2forQBQaZc` |
| `v61_2_z5dZV6g_rXM` |
| `v61_4_dQSoZh7ZjEQ` |
| `v61_6_KaVr4kbGZRE` |
| `v61_7_NQ4qXZsIrx8` |
| `v61_8_cJskrO-DANo` |

## I11C — Sonde hors domaine

Le protocole utilise :

- le checkpoint `ttnet_3rd_phase_best.pth` ;
- neuf images consécutives ;
- la dernière image comme frame cible ;
- le seuil du checkpoint : `0.05` ;
- les sorties globale et locale officielles ;
- aucun GT pour calculer l'exactitude.

Résultats globaux :

| Mesure | Valeur |
|---|---:|
| Clips | 6 |
| Frames cibles | 11 352 |
| Sorties globales valides | 0.0642 |
| Sorties raffinées valides | 0.0436 |
| Paires raffinées consécutives | 0.0254 |
| Déplacement médian | 9.13 px |
| Déplacement p90 | 21.34 px |
| Déplacement p99 | 292.97 px |
| Débit complet | 67.4 fps |

Résultats par source :

| Source | Global valide | Raffiné valide | Plus longue interruption |
|---|---:|---:|---:|
| `v61_1_M2forQBQaZc` | 0.106 | 0.082 | 400 frames |
| `v61_2_z5dZV6g_rXM` | 0.196 | 0.124 | 186 frames |
| `v61_4_dQSoZh7ZjEQ` | 0.018 | 0.006 | 1104 frames |
| `v61_6_KaVr4kbGZRE` | 0.029 | 0.018 | 424 frames |
| `v61_7_NQ4qXZsIrx8` | 0.027 | 0.024 | 362 frames |
| `v61_8_cJskrO-DANo` | 0.008 | 0.008 | 540 frames |

## Revue visuelle

Les overlays montrent :

- quelques courtes détections correctement placées sur la balle ;
- de nombreuses réponses sur les joueurs ;
- des faux positifs sur les bords de table, le filet, le sol et les panneaux ;
- des séquences valides persistantes alors que l'échange est terminé ;
- une continuité insuffisante pour constituer un suivi de balle fiable.

La validité produite par le seuil TTNet ne doit donc pas être assimilée à une
mesure de précision hors domaine.

## I11D — Contrôle de la fréquence temporelle

Pour isoler l'effet du passage de 120 fps à 50 fps, le benchmark OpenTTGames
de I10 a été rejoué avec neuf images couvrant environ 160 ms.

Offsets utilisés dans la timeline 120 fps :

    -19, -17, -14, -12, -10, -7, -5, -2, 0

Comparaison :

| Expérience | Global valide | Global R20 | Raffiné valide | Raffiné R5 | Raffiné R20 | Médiane raffinée |
|---|---:|---:|---:|---:|---:|---:|
| I10 natif 120 fps | 0.993 | 0.713 | 0.982 | 0.918 | 0.975 | 2.00 px |
| I11D simulation 50 fps | 0.993 | 0.668 | 0.974 | 0.859 | 0.968 | 2.24 px |

## Interprétation

Le changement de fréquence temporelle entraîne une baisse modérée, mais le
checkpoint reste très performant sur son domaine OpenTTGames.

L'effondrement observé dans I11C provient donc principalement du changement
de domaine visuel, et non du seul passage à 50 fps.

## Décision

Le checkpoint `ttnet_120fps` :

- reste validé comme reproduction du résultat OpenTTGames ;
- n'est pas retenu comme détecteur générique pour TTFlux ;
- ne doit pas être intégré au pipeline canonique ;
- ne justifie pas une interpolation artificielle des vidéos vers 120 fps ;
- ne doit pas être réglé ou réentraîné sans benchmark hors domaine avec GT.

La prochaine étape recommandée est un petit benchmark figé et annoté sur les
vidéos TTFlux, afin de comparer TTNet au détecteur actuel avant toute décision
de fine-tuning.
