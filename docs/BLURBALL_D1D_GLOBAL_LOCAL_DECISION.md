# BlurBall D1D — décision global / local

## Statut

La combinaison retenue pour l'évaluation finale est figée.

- Checkpoint : D1C, meilleur état à l'étape 3000
- Politique : refined_only
- Seuil global : 0.05
- Seuil local : 0.05
- Étage local : checkpoint TTNet historique
- Réentraînement local : non retenu

## Baseline D1D0

Évaluation sur les 30 échanges de validation des matchs 20 et 21.

| Mesure | Global D1C | Raffiné avec local historique |
|---|---:|---:|
| Rappel à 20 px | 0.5642066937403725 | 0.6261027867245483 |
| Précision à 20 px | 0.7932663910218547 | 0.9535082107059074 |
| F1 à 20 px | 0.6594108019639935 | 0.7558748943364327 |
| Erreur médiane valide | 8.59065 px | 2.3942 px |
| Faux positifs invisibles | 0.2181500872600349 | 0.17102966841186737 |

L'étage local récupère 603 échecs globaux et perd 161 succès globaux, soit un gain net de 442 détections correctes.

## Calibration D1D1

La politique courante refined_only avec les seuils 0.05 et 0.05 atteint :

- précision à 20 px : 0.9535082107059074
- rappel à 20 px : 0.6261027867245483
- F1 à 20 px : 0.7558748943364327
- faux positifs invisibles : 0.17102966841186737

La meilleure variante conservatrice utilise les seuils 0.03 et 0.05, pour un F1 de 0.7559574108500929.

Le gain absolu est seulement de 8.25165136602024E-05. Il est trop faible pour justifier un changement de règle après calibration sur les seuls matchs 20 et 21.

La variante maximisant le F1 réduit fortement le seuil local, mais augmente les réponses sur les frames invisibles et diminue la précision. Elle n'est pas retenue.

## Décision

Le modèle final de la phase D1 utilise :

- D1C pour ball_global_stage
- le local TTNet historique pour ball_local_stage
- refined_only
- seuil global 0.05
- seuil local 0.05
- aucun repli automatique sur la sortie globale

Cette configuration est fixée avant l'ouverture du split test.

## Protocole de test

La prochaine évaluation utilise une seule fois :

- matchs 22 à 25
- 80 échanges
- 12 777 annotations avant retrait des huit premières frames de chaque échange
- aucune modification de seuil après lecture des résultats

Les résultats du test mesureront la performance finale de cette phase. Ils ne serviront pas à réentraîner ni recalibrer le même modèle.
