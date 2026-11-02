# Rapport d'analyse expérimentale : Classification de mouvements artistiques

Ce document présente l'analyse des résultats obtenus lors de la campagne d'entraînements répertoriée dans les fichiers measures_summary.json et measures_history.json. L'objectif est d'étudier : 

- l'influence du regroupement taxonomique des classes

- du nombre d'époques et 

- de l'arrêt précoce (early stopping)

## 1. Cadre expérimental

L'ensemble des tests s'appuie sur le modèle ArtClassifier, composé d'un extracteur de caractéristiques ResNet-50 pré-entraîné suivi d'une couche dense de 512 neurones et d'un classifieur linéaire. 

Huit configurations distinctes ont été évaluées, chacune avec deux lancements indépendants (soit 16 entraînements au total) pour vérifier la stabilité des métriques :
- Dataloader simple : 117 classes brutes correspondant directement aux dossiers du jeu de données.
- Dataloader canonique : 21 macro-classes issues d'un regroupement taxonomique des mouvements artistiques.
- Nombre d'époques : 10 ou 20 époques.
- Arrêt précoce : désactivé (none) ou configuré avec une patience de 2 époques sur la perte de validation.

Les performances sont évaluées selon la perte en validation, l'exactitude micro (top-1 global), l'exactitude macro (moyenne non pondérée des rappels par classe) et le top-5.

## 2. Résultats agrégés

Le tableau ci-dessous regroupe les moyennes et écarts-types obtenus sur les deux lancements de chaque configuration (données extraites de measures_summary.json).

| Configuration (Loader / Époques / ES) | Époques effectives | Perte val. min | Perte val. finale | Micro Top-1 (%) | Macro Top-1 (%) | Top-5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| simple / 10 / none | 10.0 ± 0.0 | 1.126 ± 0.026 | 1.173 ± 0.018 | 66.12 ± 0.36 | 12.41 ± 0.26 | 91.68 ± 0.10 |
| simple / 10 / 2 | 7.5 ± 0.7 | 1.140 ± 0.015 | 1.149 ± 0.021 | 65.62 ± 0.10 | 10.04 ± 0.40 | 91.77 ± 0.26 |
| simple / 20 / none | 20.0 ± 0.0 | 1.117 ± 0.027 | 1.346 ± 0.093 | 66.00 ± 0.67 | 11.98 ± 0.01 | 92.55 ± 0.02 |
| simple / 20 / 2 | 7.0 ± 0.0 | 1.151 ± 0.008 | 1.169 ± 0.008 | 65.59 ± 0.22 | 10.32 ± 1.14 | 91.64 ± 0.47 |
| canonique / 10 / none | 10.0 ± 0.0 | 0.657 ± 0.019 | 0.786 ± 0.106 | 81.30 ± 0.15 | 52.49 ± 0.14 | 95.86 ± 0.38 |
| canonique / 10 / 2 | 7.5 ± 0.7 | 0.629 ± 0.024 | 0.653 ± 0.026 | 81.38 ± 0.47 | 52.19 ± 1.49 | 96.05 ± 0.26 |
| canonique / 20 / none | 20.0 ± 0.0 | 0.653 ± 0.016 | 0.996 ± 0.037 | 82.03 ± 0.36 | 55.54 ± 0.09 | 96.06 ± 0.06 |
| canonique / 20 / 2 | 5.5 ± 3.5 | 0.790 ± 0.194 | 1.704 ± 1.390 | 79.22 ± 1.77 | 44.24 ± 8.32 | 94.93 ± 1.62 |

## 3. Détail des 16 lancements individuels

Le suivi époque par époque consigné dans measures_history.json permet d'observer précisément la dynamique de chaque entraînement, notamment l'époque à laquelle la perte minimale a été atteinte.

| Configuration | Run | Époques faites | Époque du min(loss) | Perte val. min | Perte val. finale | Micro Top-1 (%) | Macro Top-1 (%) | Durée |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| simple / 10 / none | R1 | 10 | Ep 6 | 1.145 | 1.185 | 65.87 | 12.22 | 17.7 min |
| | R2 | 10 | Ep 6 | 1.108 | 1.160 | 66.38 | 12.59 | 17.4 min |
| simple / 10 / 2 | R1 | 8 | Ep 6 | 1.129 | 1.134 | 65.70 | 10.32 | 16.6 min |
| | R2 | 7 | Ep 5 | 1.151 | 1.164 | 65.55 | 9.75 | 12.2 min |
| simple / 20 / none | R1 | 20 | Ep 7 | 1.098 | 1.411 | 66.48 | 11.97 | 33.2 min |
| | R2 | 20 | Ep 5 | 1.136 | 1.280 | 65.52 | 11.99 | 33.9 min |
| simple / 20 / 2 | R1 | 7 | Ep 5 | 1.157 | 1.163 | 65.75 | 11.13 | 11.9 min |
| | R2 | 7 | Ep 5 | 1.146 | 1.175 | 65.43 | 9.51 | 19.1 min |
| canonique / 10 / none | R1 | 10 | Ep 5 | 0.643 | 0.861 | 81.20 | 52.59 | 15.7 min |
| | R2 | 10 | Ep 5 | 0.670 | 0.711 | 81.41 | 52.38 | 15.6 min |
| canonique / 10 / 2 | R1 | 7 | Ep 5 | 0.646 | 0.672 | 81.04 | 51.14 | 10.3 min |
| | R2 | 8 | Ep 6 | 0.612 | 0.635 | 81.71 | 53.24 | 11.7 min |
| canonique / 20 / none | R1 | 20 | Ep 7 | 0.664 | 0.969 | 81.77 | 55.60 | 33.9 min |
| | R2 | 20 | Ep 6 | 0.642 | 1.022 | 82.28 | 55.47 | 29.4 min |
| canonique / 20 / 2 | R1 | 8 | Ep 6 | 0.653 | 0.721 | 80.47 | 50.12 | 11.9 min |
| | R2 | 3 | Ep 1 | 0.927 | 2.687 | 77.97 | 38.35 | 4.4 min |

## 4. Analyse et interprétation

### Influence du regroupement de classes
La comparaison entre les deux modes de chargement met en évidence une différence structurelle majeure. Avec le chargeur simple, le modèle atteint un score micro respectable d'environ 66 %, mais son score macro s'effondre aux alentours de 11 %. Cet écart s'explique par la forte asymétrie de la distribution d'origine : la classe Social Realism représente à elle seule près de 40 % des images, alors que plusieurs dizaines de mouvements ne contiennent que quelques exemples, voire une seule image. Le réseau privilégie donc systématiquement les classes dominantes pour minimiser sa perte globale, au détriment des classes minoritaires.

À l'inverse, le regroupement en 21 classes canoniques résout cette fragmentation. La perte de validation diminue de près de moitié (passant de 1.13 à 0.63 en moyenne), le score micro atteint 81 à 82 %, et le score macro progresse de plus de 40 points pour se stabiliser entre 52 et 55 %. Le modèle apprend ainsi à discriminer de manière bien plus équilibrée les différentes périodes stylistiques.

### Comportement de l'arrêt précoce
L'activation de l'arrêt précoce avec une patience de 2 époques remplit correctement son rôle d'optimisation dans la majorité des cas : les entraînements sont coupés vers la 7e ou 8e époque, évitant la dégradation finale de la perte tout en divisant la consommation de calcul par deux sur les cycles longs.

Néanmoins, une limite opérationnelle apparaît sur le second run de la configuration canonique sur 20 époques avec arrêt anticipé. Lors de cet entraînement, la perte de validation a connu une hausse temporaire dès les premières itérations (0.927 à l'époque 1, 1.250 à l'époque 2, puis 2.687 à l'époque 3). Avec une patience fixée à seulement 2 époques, le mécanisme a interrompu le processus dès la 3e époque. Le modèle s'est retrouvé sous-entraîné, plafonnant à 38 % en macro-accuracy. Une patience de 2 s'avère donc trop sensible aux oscillations stochastiques inhérentes aux premiers batchs.

---

## 5. Bilan et préconisations

Au regard de ces résultats, la configuration la plus cohérente repose sur :
1. L'utilisation du dataloader canonique, indispensable pour garantir une représentativité équilibrée entre les mouvements artistiques.
2. Un nombre d'entraînement cible compris entre 10 et 12 époques
3. Le maintien de l'arrêt précoce, mais en portant la patience à 3 ou 4 époques afin de tolérer les variations
4. La conservation systématique des poids correspondant au point d'évaluation optimal (meilleure perte de validation) plutôt que ceux de la dernière époque.
