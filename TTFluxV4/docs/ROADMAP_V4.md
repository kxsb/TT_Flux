# Road map TTFlux V4

## Principe

TTFlux V4 est un pipeline d'analyse vidéo mesurable.

La priorité est le socle 2D : vidéo, tracking balle, overlays, exports, métriques.

## Séquence de patchs

- PATCH 001 — clean structure + CLI
- PATCH 002 — video library scanner
- PATCH 003 — minimal local UI
- PATCH 004 — analysis run folder
- PATCH 005 — baseline ball candidates
- PATCH 006 — overlay candidates + track
- PATCH 007 — metrics/report
- PATCH 008 — relink v1
- PATCH 009 — table context import
- PATCH 010 — review report

## Contrat de sortie cible

Chaque analyse doit produire :

```txt
runs/analysis_YYYYMMDD_HHMMSS/
  input_manifest.json
  video_qc.json
  raw_candidates.csv
  ball_detections.csv
  ball_track.csv
  overlay.mp4
  metrics.json
  report.html
```

## Règles

- Ne pas optimiser le top1 avant d'avoir un bon réservoir de candidats.
- La table/caméra sert au reranking, pas à un filtre dur unique.
- Si la table est incertaine, écrire `NO_TABLE_CONTEXT`.
- La 3D ne doit pas masquer un mauvais tracking 2D.
