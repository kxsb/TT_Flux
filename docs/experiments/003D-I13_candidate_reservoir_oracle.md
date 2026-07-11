# Expérience 003D-I13 — audit oracle du réservoir de candidats

## Objet

Mesurer la capacité réelle du générateur brut de petits objets mobiles à
contenir la balle sur les 450 frames du jeu de vérité terrain I12.

L'audit est réalisé indépendamment du tracker temporel et du filtre
historique v72.

## Jeu de vérité terrain

- 3 clips hors domaine
- 150 frames par clip
- 450 frames au total
- 416 frames avec balle visible
- 34 frames sans balle localisable

## Générateur évalué

Implémentation :

`ttflux.analysis.candidates.detect_frame_candidates`

Configuration canonique :

- seuil de mouvement : 18
- aire minimale : 2 px
- aire maximale : 160 px
- dimension maximale : 26 px
- remplissage minimal : 0,10
- limite canonique : 24 candidats par frame
- flou gaussien : 3

## Résultats avec limite à 24

| Rayon | Top-1 | Rappel top-1 | Oracle cap 24 | Rappel oracle |
|---|---:|---:|---:|---:|
| 5 px | 91 | 21,88 % | 110 | 26,44 % |
| 10 px | 201 | 48,32 % | 247 | 59,38 % |
| 20 px | 258 | 62,02 % | 322 | 77,40 % |
| 30 px | 282 | 67,79 % | 352 | 84,62 % |
| 50 px | 290 | 69,71 % | 371 | 89,18 % |

À 20 px, 64 frames contiennent un candidat correct dans les 24 premiers,
mais ce candidat n'est pas classé premier.

## Audit de capacité

Le générateur a également été exécuté sans troncature pratique.

- moyenne : 70,169 candidats par frame
- médiane : environ 65 candidats par frame
- percentile 95 : 155,65 candidats
- maximum : 226 candidats
- 404 frames sur 450 dépassent 24 candidats
- aucune frame n'atteint la limite étendue de 4096

Courbe de capacité à 20 px :

| Capacité | Hits | Rappel |
|---:|---:|---:|
| 1 | 258 | 62,02 % |
| 3 | 284 | 68,27 % |
| 5 | 296 | 71,15 % |
| 10 | 312 | 75,00 % |
| 24 | 322 | 77,40 % |
| 48 | 334 | 80,29 % |
| 96 | 340 | 81,73 % |
| 192 | 340 | 81,73 % |
| Réservoir complet | 340 | 81,73 % |

Le réservoir complet ajoute seulement 18 hits par rapport au cap canonique
de 24.

## Décomposition des frames visibles à 20 px

| Classe | Frames |
|---|---:|
| Candidat correct classé premier | 258 |
| Candidat correct aux rangs 2 à 24 | 64 |
| Candidat correct au-delà du rang 24 | 18 |
| Réservoir non vide sans candidat correct | 68 |
| Réservoir entièrement vide | 8 |
| Total visible | 416 |

## Résultats par domaine

| Clip | Top-1 | Rangs 2–24 | Au-delà de 24 | Échec générateur |
|---|---:|---:|---:|---:|
| `i12a_best_v61_2` | 61 | 42 | 14 | 30 |
| `i12a_red_v61_7` | 86 | 19 | 2 | 12 |
| `i12a_wide_v61_4` | 111 | 3 | 2 | 34 |

Les régimes d'échec diffèrent fortement selon le domaine :

- `best_v61_2` souffre principalement du ranking ;
- `red_v61_7` possède le meilleur générateur ;
- `wide_v61_4` classe correctement les candidats présents, mais manque
  davantage de détections.

## Contrôle de déterminisme

Une exécution mono-thread a comparé :

- le détecteur limité à 24 ;
- le préfixe des 24 premiers candidats du réservoir étendu ;
- deux exécutions successives du réservoir étendu.

Résultats :

- aucune différence entre le cap 24 et le préfixe du réservoir étendu ;
- aucune différence entre deux exécutions successives ;
- aucune égalité au rang 1 ;
- deux égalités seulement à la frontière du rang 24.

L'écart initial observé dans I13C provenait d'une erreur de définition de
la métrique cap-k : elle utilisait le rang du candidat globalement le plus
proche au lieu de tester l'existence d'un hit dans les k premiers candidats.
Cette métrique a été corrigée.

## Décision

La limite canonique reste fixée à 24 candidats par frame.

L'augmentation brute de la capacité n'est pas prioritaire :

- le passage de 24 au réservoir complet ne récupère que 18 frames ;
- 64 frames sont déjà récupérables par un meilleur ranking ;
- 76 frames nécessitent une amélioration réelle de la génération.

La prochaine étape est donc un audit des features du ranking sur les
24 candidats existants, sans modification du générateur et sans ajout de
modèle lourd.
