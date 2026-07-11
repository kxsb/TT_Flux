# Expérience 003D-F3 — validation aveugle du classifieur M0

## Statut

`M0_RETAINED_EXPERIMENTAL`

Le classifieur M0 est conservé comme baseline multivariée de
classification des tracklets.

Il n'est pas encore considéré comme moteur balle canonique.

## Modèle figé

- Entraînement : `003D-F0 Wave 1`
- Seuil de décision : `0.5`
- SHA-256 : `1789844aacf08be5e325a9ea3af5879cdd6acbfde15078461093b51c8821e368`
- Modèle : régression logistique standardisée
- Bibliothèque : scikit-learn `1.9.0`
- Python : `3.13.9`

Le modèle a été figé avant l'import des annotations Wave 2.

## Périmètre évalué

Classification binaire d'un tracklet :

- `ball`
- `not_ball`

Les annotations `uncertain` sont exclues des métriques.

La signification actuelle de `ball` est :

> le tracklet contient une portion de balle suffisamment identifiable.

Elle ne garantit pas encore que le tracklet suive exclusivement la balle
du début à la fin.

## Résultat Wave 1 — validation inter-sources

- TP=29, FP=8, TN=49, FN=9, précision=0.784, rappel=0.763, F1=0.773
- accuracy=0.821
- balanced accuracy=0.811
- ROC AUC=0.896

## Résultat Wave 2 — validation aveugle

Nombre de tracklets évaluables : `113`

### Baseline R0

- TP=46, FP=2, TN=31, FN=34, précision=0.958, rappel=0.575, F1=0.719
- accuracy=0.681
- balanced accuracy=0.757

### Classifieur M0 figé

- TP=74, FP=9, TN=24, FN=6, précision=0.892, rappel=0.925, F1=0.908
- accuracy=0.867
- balanced accuracy=0.826
- ROC AUC=0.900

## Résultat par source

| Source | N | TP | FP | TN | FN | Précision | Rappel | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opentt_game4 | 37 | 23 | 2 | 9 | 3 | 0.920 | 0.885 | 0.902 |
| v60_03 | 32 | 23 | 1 | 6 | 2 | 0.958 | 0.920 | 0.939 |
| v60_08 | 44 | 28 | 6 | 9 | 1 | 0.824 | 0.966 | 0.889 |

## Interprétation

Le classifieur M0 améliore fortement le rappel par rapport à R0 sans
effondrement complet de la précision.

Le score F1 dépasse 0,90 sur la validation aveugle globale.

La précision globale est de 0,892, légèrement sous le seuil cible de 0,90.

## Limites connues

1. Corpus majoritairement ralenti ou temporellement ralenti.
2. Certains tracklets positifs changent d'identité en cours de trajectoire.
3. La pureté temporelle du tracklet n'est pas encore annotée.
4. La couverture frame par frame de la balle n'est pas encore mesurée.
5. Seulement six familles de fenêtres ont servi aux deux premières vagues.
6. Le modèle dépend des caractéristiques produites par le détecteur actuel.

## Décision

- Conserver M0 sans le réentraîner avec Wave 2.
- Conserver le seuil de probabilité à 0,5.
- Ne pas intégrer encore le modèle dans le moteur canonique.
- Effectuer une troisième vague exclusivement à vitesse réelle.
- Ajouter ultérieurement une annotation distincte de pureté :
  `clean`, `contaminated`, `identity_switch`.

## Critère du prochain jalon

Sur un corpus `realtime_only` indépendant :

- F1 global >= 0,90 ;
- précision >= 0,90 ;
- rappel >= 0,90 ;
- aucune source avec F1 < 0,75 ;
- modèle figé avant annotation.
