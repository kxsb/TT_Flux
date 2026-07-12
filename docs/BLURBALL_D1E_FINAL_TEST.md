# BlurBall D1E — évaluation finale sur le split test

## Statut

Le split test BlurBall a été ouvert une seule fois après le gel de la
politique d'inférence D1D.

Les résultats D1E ne doivent pas servir à modifier les seuils ou à réentraîner
le même modèle D1.

## Protocole gelé

- Matchs test : 22 à 25
- Échanges : 80
- Frames évaluées par modèle : 12137
- Politique : refined_only
- Seuil global : 0.05
- Seuil local : 0.05
- Distance de succès : 20 px
- Étage global : checkpoint D1C, meilleure étape 3000
- Étage local : étage TTNet historique inchangé

## Résultat principal

| Mesure | TTNet original | TTNet D1 final | Delta |
|---|---:|---:|---:|
| Précision à 20 px | 0.9022222222222223 | 0.8201598579040853 | -0.08206236431813696 |
| Rappel à 20 px | 0.017723066177754498 | 0.48376113148245153 | 0.46603806530469705 |
| F1 à 20 px | 0.03476325027827725 | 0.6085667215815486 | 0.5738034713032714 |
| Erreur médiane valide | 2.22825 px | 4.5616 px | 2.3333500000000003 px |
| Faux positifs invisibles | 0.0014641288433382138 | 0.13030746705710103 | 0.12884333821376281 |

Le rappel brut du modèle D1 final atteint 0.5928060066352366.

## Couverture

| Mesure | TTNet original | TTNet D1 final |
|---|---:|---:|
| Prédictions valides | 225 | 6756 |
| Frames visibles acceptées | 224 | 6667 |
| Frames invisibles acceptées | 1 | 89 |

La précision élevée du modèle original correspond à une couverture extrêmement
faible. Il ne produit que 225 prédictions valides sur
12137 frames évaluées.

Le modèle D1 final produit 6756 prédictions valides et augmente
le rappel de 0.46603806530469705.

## Généralisation validation vers test

| Mesure | Validation | Test | Delta test - validation |
|---|---:|---:|---:|
| Précision à 20 px | 0.95350821070590741 | 0.8201598579040853 | -0.133348352801822 |
| Rappel à 20 px | 0.62610278672454833 | 0.48376113148245153 | -0.142341655242097 |
| F1 à 20 px | 0.75587489433643273 | 0.6085667215815486 | -0.147308172754884 |
| Erreur médiane valide | 2.3942315865428809 px | 4.5616 px | 2.16736841345712 px |
| Faux positifs invisibles | 0.17102966841186737 | 0.13030746705710103 | -0.0407222013547663 |

La baisse validation-test confirme une sensibilité résiduelle au domaine.
Elle ne remet pas en cause le gain important obtenu par rapport au checkpoint
TTNet d'origine.

## Interprétation

D1 constitue une adaptation réussie de l'étage global TTNet au domaine
BlurBall.

Le local TTNet historique reste suffisamment compatible avec le nouvel étage
global pour améliorer la précision spatiale et le F1 final.

Les principales limites observées sont :

- rappel final inférieur au rappel brut ;
- faux positifs sur les annotations invisibles ;
- variabilité importante entre certains matchs et échanges ;
- comportement plus faible sur une partie du match 25.

Ces constats doivent alimenter une nouvelle phase expérimentale distincte,
avec un nouveau protocole de validation. Ils ne doivent pas être utilisés pour
recalibrer rétroactivement D1.

## Artefacts locaux non versionnés

- runs/dataset_benchmark/d1e_blurball_final_test/summary.json
- runs/dataset_benchmark/d1e_blurball_final_test/test_predictions.csv
- runs/dataset_benchmark/d1e_blurball_final_test/test_by_rally.csv
