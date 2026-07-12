# Clôture du chantier refactor TTFlux

Statut officiel :

    REFACTOR_CHANTIER_CLOSED

Date de clôture : 12 juillet 2026.

Tag de référence :

    refactor-closed-2026-07-12

## Périmètre clôturé

Le chantier a couvert :

- le gel de l'API publique du tracking ;
- la séparation des contrats et artefacts ;
- l'extraction des candidats, scorers et pistes temporelles ;
- le déplacement de l'orchestration vers `ttflux.pipeline` ;
- l'extraction des états, du stockage, des clips, des rapports,
  des artefacts, de la maintenance et du catalogue ;
- la conservation des imports historiques nécessaires ;
- l'introduction des contrats purs de scène 2D ;
- l'intégration par fast-forward dans `main`.

Le moteur de balle conserve le statut :

    BALL_TRACKING_BASELINE_FROZEN_EXPLORATORY

La clôture du refactor ne constitue pas une validation de sa
précision.

## Audit post-refactor

L'audit a vérifié :

- 92 fichiers Python ;
- 16 fichiers source UTF-8 avec BOM correctement reconnus ;
- aucune erreur de parsing ;
- aucun cache Python suivi par Git ;
- les alias et imports historiques de compatibilité ;
- le chargement paresseux de l'API publique du tracking ;
- le point d'entrée `python -m ttflux`.

Aucune suppression de code runtime n'a été retenue.

Les modules historiques encore présents assurent une compatibilité
explicitement testée. Ils ne sont pas considérés comme du
dead code.

## Smoke tests fonctionnels

La clôture a été précédée des contrôles
suivants :

- 93 tests automatiques réussis ;
- CLI TTFlux 0.1.4 fonctionnelle ;
- commande `ttflux doctor` au statut prêt ;
- FFmpeg et FFprobe disponibles ;
- import des API publiques `pipeline`, `tracking` et `scene` ;
- exécution complète d'un run synthétique isolé ;
- production et vérification de 11 artefacts ;
- lecture du run depuis le catalogue ;
- suppression contrôlée du run ;
- préservation de la vidéo source ;
- démarrage réel du serveur Uvicorn ;
- validation de `/health`, `/api/videos`, `/api/runs`
  et de la page d'accueil ;
- nettoyage des caches Python locaux ;
- dépôt Git propre après exécution.

## Branches conservées

- `main` : branche canonique de développement ;
- `refactor/repo-layout` : état final du chantier avant clôture ;
- `ttflux-v0-clean` : baseline historique immuable ;
- `archive/ball-tracker-pre-scene-2026-07-12` : archive distante.

## Suite du projet

Les développements fonctionnels reprennent depuis `main`.

L'étape active est R5B : persistance optionnelle du contexte de
scène.

R5B doit être développé sur une branche dédiée et
ne doit pas rouvrir le tracking de balle sans nouvelle mesure
indépendante.
