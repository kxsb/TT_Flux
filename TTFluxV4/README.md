# TTFlux V4

TTFlux V4 est la nouvelle base propre du projet.

Objectif immédiat : construire un pipeline d'analyse vidéo mesurable, pas empiler des prototypes.

Priorités V4 :

1. naviguer dans les vidéos brutes ;
2. lancer une analyse sur une vidéo ;
3. produire des sorties standardisées ;
4. générer des overlays de contrôle ;
5. mesurer les échecs ;
6. préparer ensuite table, balle, pose et événements.

Règle produit :

> La 3D, le spin et la biomécanique ne démarrent pas tant que le socle 2D balle/table/overlay/évaluation n'est pas stable.

## Structure

```txt
TTFluxV4/
  app/
  data/
  docs/
  runs/
  scripts/
  src/ttflux/
  tests/
```

## Commandes prévues

```powershell
cd TTFluxV4
python -m ttflux --help
python -m ttflux smoke
```

## Statut

PATCH 001 : socle vierge, CLI minimale, contrats de données, dossier de road map.
