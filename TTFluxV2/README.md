# TTFlux V2

Nouvelle base propre.

Principe de conception :
1. Interface d'analyse locale d'abord.
2. Analyse environnement : table, caméra, score, joueurs.
3. Tracking balle.
4. Moteur d'entraînement PyTorch.
5. Dataset builders.
6. Interface d'analyse tactique.

État hérité TTFlux V1 :
- Table / caméra : récupérer les artefacts validés 005C6A à 005C9B.
- Tracking balle : récupérer comme référence, pas comme code direct.
- Diagnostic important : les gros gaps venaient surtout d'un réservoir candidat vide, pas du relink.

Lancement UI :
PowerShell :
  cd C:\Users\micka\Desktop\développement\ping\TTFluxV2
  powershell -ExecutionPolicy Bypass -File .\scripts\run_ui.ps1

Puis ouvrir :
  http://127.0.0.1:8787
