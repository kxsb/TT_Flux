# Entraînement BlurBall

Ces scripts reproduisent la première adaptation de TTNet au corpus BlurBall.

## Ordre d'exécution

1. `build_blurball_manifest_b1.py`
2. `run_blurball_micro_overfit_d1a.py`
3. `run_blurball_global_pilot_d1b.py`
4. `train_blurball_global_full_d1c.py`

## Splits gelés

- Train : matchs `00` à `19`
- Validation : matchs `20` et `21`
- Test : matchs `22` à `25`

Les scripts D1A à D1C ne doivent jamais utiliser le split test.

## État après D1C

D1C entraîne uniquement `ball_global_stage`.

Les modules suivants restent ceux du checkpoint TTNet d'origine :

- `ball_local_stage`
- `events_spotting`
- `segmentation`

Les checkpoints et rapports volumineux sont produits sous `runs/` et ne sont pas versionnés.

## Évaluation finale D1E

5. `evaluate_blurball_final_test_d1e.py`

Le test final utilise les matchs 22 à 25 avec la politique figée
`refined_only` et les seuils 0.05 / 0.05.

Le résultat synthétique versionné est disponible dans
`blurball_d1_final_test_result.json`.

Le split test D1 est désormais ouvert et ne doit plus servir à recalibrer
le modèle D1.
