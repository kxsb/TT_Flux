# Expérience 003D-I14FV — validation visuelle de l’occupation et des trajectoires

## Objet

Compléter les audits quantitatifs I14C–I14F par deux sorties visuelles :

- carte globale d’occupation des candidats ;
- comparaison vidéo entre la trajectoire GT et le reranking temporel.

Ces outils sont des scripts de diagnostic. Ils ne constituent pas un
tracker canonique ni un modèle de production.

## Carte d’occupation

Sur `i12a_wide_v61_4`, la carte d’occupation globale se concentre
principalement autour des deux joueurs.

La superposition montre que :

- les faux positifs temporels sont fortement regroupés dans ces zones ;
- les candidats balle traversent davantage des zones moins persistantes ;
- la dérive R014 correspond à un verrouillage prolongé sur le joueur de
  gauche.

Cela confirme visuellement le résultat de I14F : l’occupation contient
une information utile pour différencier la balle des distracteurs joueur.

Cette information ne doit toutefois être utilisée que comme une pénalité
douce. La balle peut passer devant le corps, la main ou la raquette.

## Vidéo de trajectoire

La vidéo compare :

- cyan : position GT ;
- vert : prédiction correcte ;
- rouge : prédiction erronée ;
- jaune : distance entre la prédiction erronée et la GT.

Elle confirme que le pipeline sait produire plusieurs séquences
correctes et temporellement cohérentes.

Elle confirme également que les principales erreurs correspondent à une
acquisition erronée d’un joueur, ensuite stabilisée par la continuité
temporelle.

## Limitation connue du rendu I14FV2

Le trail visuel peut relier deux prédictions séparées par une ou plusieurs
frames d’abstention.

Certaines longues diagonales visibles ne représentent donc pas une
trajectoire continue réellement prédite.

Une future variante devra interrompre le trail lors :

- d’une abstention ;
- d’un gap supérieur à la limite admise ;
- d’une réacquisition de piste.

Cette limitation concerne uniquement le rendu vidéo, pas les métriques
I14C.

## Décision

Les scripts sont conservés comme outils de diagnostic reproductibles.

Les sorties MP4 et PNG restent dans `runs/` et ne sont pas versionnées.

La fondation à conserver est constituée par :

- le corpus GT gelé ;
- le réservoir de candidats ;
- les audits oracle ;
- les protocoles leave-one-clip-out ;
- les overlays et galeries ;
- les métriques de couverture, précision, rappel et longueur de runs.

Les modèles expérimentaux I14B–I14F ne deviennent pas encore le modèle
canonique.
