# Rapport d'analyse expérimentale : Classification de mouvements artistiques (ConvNeXt Tiny)

Ce document présente l'analyse des résultats obtenus lors de la campagne d'entraînements du modèle ArtClassifierTinyBased, répertoriée dans les fichiers measures_summary.json et measures_history.json. Il reprend le protocole du rapport du 1er novembre (ResNet-50) afin de comparer les deux extracteurs. L'objectif est d'étudier :

- l'influence du regroupement taxonomique des classes

- du nombre d'époques et

- de l'arrêt précoce (early stopping)

- de l'extracteur de caractéristiques (ConvNeXt Tiny face à ResNet-50)

## 1. Cadre expérimental

L'ensemble des tests s'appuie sur le modèle ArtClassifierTinyBased, composé d'un extracteur de caractéristiques ConvNeXt Tiny pré-entraîné sur ImageNet (768 caractéristiques en sortie), suivi d'une couche dense de 512 neurones (BatchNorm, GELU, Dropout 0.4) et d'un classifieur linéaire. L'ensemble du réseau est affiné avec Adam (lr = 1e-4), sur des images de 288 × 288 par lots de 16, sur une carte RTX 3060 Ti.

Huit configurations distinctes ont été évaluées, chacune avec deux lancements indépendants (soit 16 entraînements au total) pour vérifier la stabilité des métriques :
- Dataloader simple : 117 classes brutes correspondant directement aux dossiers du jeu de données.
- Dataloader canonique : 21 macro-classes issues d'un regroupement taxonomique des mouvements artistiques.
- Nombre d'époques : 10 ou 20 époques.
- Arrêt précoce : désactivé (none) ou configuré avec une patience de 2 époques sur la perte de validation.

Les performances sont évaluées selon la perte en validation, l'exactitude micro (top-1 global), l'exactitude macro (moyenne non pondérée des rappels par classe) et le top-5. Comme dans le rapport précédent, les scores micro, macro et top-5 sont les meilleures valeurs atteintes au cours de l'entraînement.

## 2. Résultats agrégés

Le tableau ci-dessous regroupe les moyennes et écarts-types obtenus sur les deux lancements de chaque configuration (données extraites de measures_summary.json).

| Configuration (Loader / Époques / ES) | Époques effectives | Perte val. min | Perte val. finale | Micro Top-1 (%) | Macro Top-1 (%) | Top-5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| simple / 10 / none | 10.0 ± 0.0 | 1.073 ± 0.023 | 1.139 ± 0.032 | 67.54 ± 0.09 | 11.75 ± 0.14 | 92.94 ± 0.12 |
| simple / 10 / 2 | 6.5 ± 0.7 | 1.087 ± 0.021 | 1.151 ± 0.053 | 66.80 ± 0.52 | 10.63 ± 0.21 | 91.58 ± 0.14 |
| simple / 20 / none | 20.0 ± 0.0 | 1.101 ± 0.013 | 1.321 ± 0.021 | 66.42 ± 0.09 | 11.82 ± 0.76 | 93.04 ± 0.40 |
| simple / 20 / 2 | 8.0 ± 1.4 | 1.090 ± 0.001 | 1.113 ± 0.004 | 66.86 ± 0.02 | 11.16 ± 0.80 | 92.34 ± 0.48 |
| canonique / 10 / none | 10.0 ± 0.0 | 0.623 ± 0.015 | 0.726 ± 0.066 | 82.37 ± 0.85 | 54.03 ± 1.87 | 96.33 ± 0.02 |
| canonique / 10 / 2 | 7.5 ± 0.7 | 0.631 ± 0.027 | 0.644 ± 0.036 | 81.95 ± 0.47 | 52.45 ± 1.05 | 95.91 ± 0.58 |
| canonique / 20 / none | 20.0 ± 0.0 | 0.641 ± 0.023 | 1.039 ± 0.002 | 83.26 ± 0.19 | 58.04 ± 1.26 | 96.17 ± 0.00 |
| canonique / 20 / 2 | 6.5 ± 2.1 | 0.621 ± 0.045 | 0.690 ± 0.039 | 81.93 ± 0.53 | 51.65 ± 0.81 | 95.77 ± 0.09 |

## 3. Détail des 16 lancements individuels

Le suivi époque par époque consigné dans measures_history.json permet d'observer précisément la dynamique de chaque entraînement, notamment l'époque à laquelle la perte minimale a été atteinte.

| Configuration | Run | Époques faites | Époque du min(loss) | Perte val. min | Perte val. finale | Micro Top-1 (%) | Macro Top-1 (%) | Durée |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| simple / 10 / none | R1 | 10 | Ep 4 | 1.089 | 1.117 | 67.48 | 11.65 | 24.4 min |
| | R2 | 10 | Ep 6 | 1.057 | 1.161 | 67.60 | 11.84 | 23.1 min |
| simple / 10 / 2 | R1 | 7 | Ep 5 | 1.072 | 1.114 | 67.16 | 10.77 | 16.0 min |
| | R2 | 6 | Ep 4 | 1.101 | 1.188 | 66.43 | 10.48 | 13.7 min |
| simple / 20 / none | R1 | 20 | Ep 5 | 1.110 | 1.306 | 66.48 | 12.36 | 47.2 min |
| | R2 | 20 | Ep 8 | 1.092 | 1.336 | 66.36 | 11.29 | 47.4 min |
| simple / 20 / 2 | R1 | 9 | Ep 7 | 1.089 | 1.110 | 66.85 | 10.59 | 22.4 min |
| | R2 | 7 | Ep 5 | 1.091 | 1.116 | 66.87 | 11.72 | 17.3 min |
| canonique / 10 / none | R1 | 10 | Ep 5 | 0.634 | 0.773 | 81.77 | 52.71 | 19.3 min |
| | R2 | 10 | Ep 7 | 0.612 | 0.680 | 82.98 | 55.35 | 19.2 min |
| canonique / 10 / 2 | R1 | 7 | Ep 5 | 0.650 | 0.670 | 81.62 | 51.70 | 14.1 min |
| | R2 | 8 | Ep 6 | 0.613 | 0.618 | 82.28 | 53.19 | 16.5 min |
| canonique / 20 / none | R1 | 20 | Ep 7 | 0.625 | 1.040 | 83.13 | 58.93 | 42.1 min |
| | R2 | 20 | Ep 5 | 0.657 | 1.038 | 83.40 | 57.15 | 42.4 min |
| canonique / 20 / 2 | R1 | 5 | Ep 3 | 0.653 | 0.663 | 81.56 | 51.07 | 9.8 min |
| | R2 | 8 | Ep 6 | 0.589 | 0.717 | 82.31 | 52.22 | 15.6 min |

## 4. Analyse et interprétation

### Influence du regroupement de classes
Le constat du rapport précédent se confirme avec ConvNeXt Tiny. Avec le chargeur simple, le score micro plafonne entre 66 et 68 %, et le score macro reste entre 10 et 12 %. La distribution brute est très asymétrique : la classe Social realism regroupe 8 032 des 20 623 images (38,9 %), tandis que 51 des 117 dossiers contiennent moins de 10 images, et trois n'en contiennent qu'une seule. Le réseau favorise les classes dominantes et ignore presque entièrement les autres.

Le regroupement en 21 classes canoniques fait passer la perte de validation minimale d'environ 1.09 à 0.63, le score micro à 82-83 % et le score macro à 52-58 %, soit un gain de plus de 40 points.

### Perte de validation et précision évoluent différemment
Dans tous les lancements, la perte de validation minimale est atteinte tôt, entre la 3e et la 8e époque. Sans arrêt précoce, elle remonte ensuite nettement : sur 20 époques, elle finit à 1.04 en canonique (contre 0.64 au minimum) et à 1.32 en simple.

En revanche, la précision continue de progresser après ce minimum. En canonique sur 20 époques sans arrêt, le meilleur score macro est atteint à l'époque 13 (R1, 58.93 %) et à l'époque 10 (R2, 57.15 %), alors que la perte minimale se situe aux époques 7 et 5. À l'époque du minimum de perte, le macro de R1 n'était que de 52.27 %. La perte remonte parce que le modèle devient trop confiant sur les images qu'il classe mal, ce qui est fortement pénalisé par l'entropie croisée, même si le nombre de bonnes réponses augmente encore.

C'est pourquoi la configuration canonique / 20 / none obtient le meilleur score macro de la campagne (58.04 %), malgré une perte finale élevée. Un arrêt précoce piloté par la perte coupe l'entraînement avant ce point.

Ces meilleurs scores sont toutefois sélectionnés sur l'ensemble de validation lui-même, faute d'ensemble de test séparé : ils sont donc légèrement optimistes.

### Comportement de l'arrêt précoce
Avec une patience de 2, les entraînements s'arrêtent entre la 5e et la 9e époque. Le gain de temps est important : les deux lancements canonique / 20 / 2 ont duré 25.4 minutes au total, contre 84.5 minutes pour canonique / 20 / none (−70 %).

Le phénomène observé par le rapport précédent réapparaît, de façon moins marquée : sur le run R1 de canonique / 20 / 2, la perte passe de 0.653 (époque 3) à 0.888 (époque 4), puis redescend à 0.663 (époque 5), sans repasser sous le minimum. Le mécanisme s'arrête donc à l'époque 5, avec un score macro de 51.07 %, contre 57-59 % pour les lancements qui ont continué.

À noter : les valeurs « finales » des configurations avec arrêt précoce correspondent à la dernière époque exécutée. Le modèle effectivement conservé est celui de l'époque du minimum de perte, rechargé à l'arrêt.

## 5. Comparaison avec ResNet-50

Le tableau suivant met en regard les meilleures valeurs moyennes de ce rapport et celles du rapport du 1er novembre (ArtClassifier, ResNet-50).

| Configuration | Perte val. min ResNet-50 | Perte val. min ConvNeXt | Micro ResNet-50 (%) | Micro ConvNeXt (%) | Macro ResNet-50 (%) | Macro ConvNeXt (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| simple / 10 / none | 1.126 | 1.073 | 66.12 | 67.54 | 12.41 | 11.75 |
| simple / 10 / 2 | 1.140 | 1.087 | 65.62 | 66.80 | 10.04 | 10.63 |
| simple / 20 / none | 1.117 | 1.101 | 66.00 | 66.42 | 11.98 | 11.82 |
| simple / 20 / 2 | 1.151 | 1.090 | 65.59 | 66.86 | 10.32 | 11.16 |
| canonique / 10 / none | 0.657 | 0.623 | 81.30 | 82.37 | 52.49 | 54.03 |
| canonique / 10 / 2 | 0.629 | 0.631 | 81.38 | 81.95 | 52.19 | 52.45 |
| canonique / 20 / none | 0.653 | 0.641 | 82.03 | 83.26 | 55.54 | 58.04 |
| canonique / 20 / 2 | 0.790 | 0.621 | 79.22 | 81.93 | 44.24 | 51.65 |

ConvNeXt Tiny obtient une perte plus faible dans 7 configurations sur 8, et un score micro supérieur dans toutes les configurations (de +0.4 à +1.4 point, hors canonique / 20 / 2). En canonique, le gain macro va de +0.3 à +2.5 points ; en simple, les scores macro des deux modèles restent équivalents, autour de 11 %. L'écart plus important sur canonique / 20 / 2 vient surtout du run ResNet-50 interrompu à la 3e époque, et non d'une supériorité du modèle. Avec seulement deux lancements par configuration, ces écarts restent du même ordre que la variabilité entre runs : l'avantage de ConvNeXt est cohérent mais modeste.

Une époque ConvNeXt dure environ 2.0 minutes en canonique et 2.4 minutes en simple, contre environ 1.6 et 1.8 minute pour ResNet-50. Cette comparaison n'est valable que si les deux campagnes ont tourné sur le même matériel.

---

## 6. Bilan et préconisations

Au regard de ces résultats, la configuration la plus cohérente repose sur :
1. L'utilisation du dataloader canonique, dont l'intérêt est confirmé avec un second extracteur.
2. ConvNeXt Tiny comme extracteur, pour un gain faible mais constant par rapport à ResNet-50. Davantage de lancements seraient nécessaires pour confirmer l'écart.
3. Un nombre d'époques cible compris entre 10 et 15 : sans arrêt précoce, le meilleur score macro est atteint entre la 8e et la 13e époque en canonique.
4. Le maintien de l'arrêt précoce, avec une patience de 3 à 4 époques, comme préconisé dans le rapport précédent. Il serait également pertinent de le piloter sur le score macro plutôt que sur la perte, puisque celle-ci remonte avant que la précision n'atteigne son maximum.
5. La conservation des poids correspondant au meilleur score macro de validation, et l'ajout d'un ensemble de test séparé pour mesurer les performances sans biais de sélection.
6. La prise en compte du déséquilibre des classes (pondération de la perte ou rééquilibrage des lots), le score macro plafonnant autour de 55 % alors que Réalisme Social représente encore 48 % des images après regroupement.
