# Rapport d'analyse expérimentale : Classification de mouvements artistiques (CustomArtCNN)

Ce document présente l'analyse des résultats obtenus lors de la campagne d'entraînements répertoriée dans les fichiers `measures_summary.json` et `measures_history.json`. L'objectif est d'étudier : 

- l'influence du regroupement taxonomique des classes
- du nombre d'époques et 
- de l'arrêt précoce (early stopping)

## 1. Cadre expérimental

L'ensemble des tests s'appuie sur le modèle `CustomArtCNN`, une architecture convolutive personnalisée entraînée de bout en bout sur la tâche de classification artistique.

Huit configurations distinctes ont été évaluées, chacune avec un lancement (`num_runs: 1`, soit 8 entraînements au total) :
- **Dataloader simple** : 117 classes brutes correspondant directement aux dossiers du jeu de données.
- **Dataloader canonique** : 21 macro-classes issues d'un regroupement taxonomique des mouvements artistiques.
- **Nombre d'époques** : 10 ou 20 époques.
- **Arrêt précoce** : désactivé (`none`) ou configuré avec une patience de 2 époques (`2`) sur la perte de validation.

Les performances sont évaluées selon la perte en validation, l'exactitude micro (top-1 global), l'exactitude macro (moyenne non pondérée des rappels par classe) et le top-5.

## 2. Résultats agrégés

Le tableau ci-dessous regroupe les performances obtenues pour chaque configuration (données extraites de `measures_summary.json`). Un seul lancement ayant été effectué par configuration, l'écart-type est nul ($\pm 0.0$). Les valeurs de précision correspondent aux meilleurs scores atteints durant le cycle.

| Configuration (Loader / Époques / ES) | Époques effectives | Perte val. min | Perte val. finale | Micro Top-1 (%) | Macro Top-1 (%) | Top-5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| simple / 10 / none | $10.0 \pm 0.0$ | $1.684 \pm 0.000$ | $1.684 \pm 0.000$ | $56.99 \pm 0.00$ | $4.60 \pm 0.00$ | $81.11 \pm 0.00$ |
| simple / 10 / 2 | $10.0 \pm 0.0$ | $1.628 \pm 0.000$ | $1.628 \pm 0.000$ | $57.13 \pm 0.00$ | $4.56 \pm 0.00$ | $81.45 \pm 0.00$ |
| simple / 20 / none | $20.0 \pm 0.0$ | $1.580 \pm 0.000$ | $1.580 \pm 0.000$ | $58.58 \pm 0.00$ | $5.06 \pm 0.00$ | $82.51 \pm 0.00$ |
| simple / 20 / 2 | $7.0 \pm 0.0$ | $1.803 \pm 0.000$ | $1.821 \pm 0.000$ | $55.22 \pm 0.00$ | $4.32 \pm 0.00$ | $79.40 \pm 0.00$ |
| canonique / 10 / none | $10.0 \pm 0.0$ | $1.258 \pm 0.000$ | $1.258 \pm 0.000$ | $65.74 \pm 0.00$ | $17.68 \pm 0.00$ | $86.81 \pm 0.00$ |
| canonique / 10 / 2 | $10.0 \pm 0.0$ | $1.209 \pm 0.000$ | $1.209 \pm 0.000$ | $66.07 \pm 0.00$ | $19.20 \pm 0.00$ | $87.17 \pm 0.00$ |
| canonique / 20 / none | $20.0 \pm 0.0$ | $1.184 \pm 0.000$ | $1.290 \pm 0.000$ | $67.46 \pm 0.00$ | $20.78 \pm 0.00$ | $87.99 \pm 0.00$ |
| canonique / 20 / 2 | $6.0 \pm 0.0$ | $1.419 \pm 0.000$ | $1.539 \pm 0.000$ | $64.08 \pm 0.00$ | $16.54 \pm 0.00$ | $84.97 \pm 0.00$ |

## 3. Détail des 8 lancements individuels

Le suivi époque par époque consigné dans `measures_history.json` permet d'observer précisément la dynamique de chaque entraînement, notamment l'époque à laquelle la perte minimale a été atteinte ainsi que la durée totale d'exécution.

| Configuration | Run | Époques faites | Époque du min(loss) | Perte val. min | Perte val. finale | Micro Top-1 max (%) | Macro Top-1 max (%) | Durée |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| simple / 10 / none | R1 | 10 | Ep 10 | 1.684 | 1.684 | 56.99 | 4.60 | 34.5 min |
| simple / 10 / 2 | R1 | 10 | Ep 10 | 1.628 | 1.628 | 57.13 | 4.56 | 34.1 min |
| simple / 20 / none | R1 | 20 | Ep 20 | 1.580 | 1.580 | 58.58 | 5.06 | 91.2 min* |
| simple / 20 / 2 | R1 | 7 | Ep 5 | 1.803 | 1.821 | 55.22 | 4.32 | 23.8 min |
| canonique / 10 / none | R1 | 10 | Ep 10 | 1.258 | 1.258 | 65.74 | 17.68 | 28.4 min |
| canonique / 10 / 2 | R1 | 10 | Ep 10 | 1.209 | 1.209 | 66.07 | 19.20 | 28.3 min |
| canonique / 20 / none | R1 | 20 | Ep 15 | 1.184 | 1.290 | 67.46 | 20.78 | 87.4 min* |
| canonique / 20 / 2 | R1 | 6 | Ep 4 | 1.419 | 1.539 | 64.08 | 16.54 | 17.1 min |

*\*Note sur la durée : Les entraînements de 20 époques sans arrêt précoce présentent chacun une époque anormalement longue dans l'historique (1612 s à l'époque 5 pour `simple/20/none` et 2012 s à l'époque 3 pour `canonique/20/none`, contre ~170 à 205 s par époque en moyenne), probablement liée à un ralentissement ponctuel du système ou des entrées/sorties.*

## 4. Analyse et interprétation

### Influence du regroupement de classes
La comparaison entre les deux modes de chargement met en évidence une différence structurelle majeure, particulièrement marquée sur cette architecture `CustomArtCNN`. Avec le chargeur `simple` (117 classes), le modèle atteint un score micro modéré compris entre 55 % et 58.6 %, mais son score macro s'effondre totalement entre 4.3 % et 5.1 %. Cet écart s'explique par la forte asymétrie de la distribution d'origine : le réseau se concentre quasi exclusivement sur les quelques classes ultra-majoritaires pour réduire sa perte globale, ignorant la grande majorité des mouvements artistiques minoritaires.

À l'inverse, le regroupement en 21 classes `canoniques` atténue significativement cette fragmentation. La perte de validation minimale diminue nettement (descendant jusqu'à 1.184 contre 1.580 au mieux en mode simple), le score micro gagne près de 9 points pour atteindre 67.46 %, et le score macro est multiplié par quatre, dépassant les 20.7 %. Bien que l'exactitude macro reste modeste dans l'absolu (caractéristique d'un réseau entraîné sans transfert d'apprentissage lourd face à des classes encore déséquilibrées), le modèle parvient à discriminer un spectre beaucoup plus large de courants artistiques.

### Dynamique d'apprentissage et nombre d'époques
Contrairement à un modèle pré-entraîné qui converge en quelques itérations, l'architecture `CustomArtCNN` présente une courbe d'apprentissage plus lente et progressive :
- Sur les cycles de **10 époques** (`simple` comme `canonique`), la perte minimale est systématiquement atteinte à la toute dernière époque (Ep 10), prouvant que le modèle n'a pas encore fini de converger.
- Le passage à **20 époques** sans arrêt précoce permet de poursuivre cette progression : en configuration `simple`, la perte continue de baisser jusqu'à l'époque 20 (1.580). En configuration `canonique`, la perte atteint son minimum à l'époque 15 (1.184) tandis que les scores Micro Top-1 (67.46 %) et Macro Top-1 (20.78 %) continuent de progresser jusqu'aux époques 17 et 20.

### Comportement de l'arrêt précoce
L'analyse met en lumière un effet contre-productif d'un arrêt précoce configuré avec une patience trop faible (`early_stop = 2`) sur ce modèle :
- Sur les configurations à **10 époques**, les fluctuations de la perte de validation n'ont pas déclenché d'arrêt anticipé (les rares remontées n'ayant jamais excédé une époque consécutive), permettant aux modèles d'aller au bout des 10 époques.
- En revanche, sur les configurations à **20 époques**, le bruit stochastique a provoqué une interruption prématurée. Pour `simple/20/2`, un minimum local à l'époque 5 (1.803) suivi de deux légères hausses (1.856 puis 1.821) a coupé l'entraînement dès l'époque 7. De même, pour `canonique/20/2`, un creux à l'époque 4 (1.419) suivi d'une stagnation à 1.539 aux époques 5 et 6 a stoppé le processus à l'époque 6.

Dans les deux cas, le modèle s'est retrouvé sous-entraîné, perdant plus de 3 points de Micro Top-1 et plus de 4 points de Macro Top-1 (16.54 % contre 20.78 % en canonique) par rapport à son équivalent laissé libre sur 20 époques.

---

## 5. Bilan et préconisations

Au regard des performances spécifiques du modèle `CustomArtCNN`, la stratégie d'entraînement optimale repose sur :
1. **L'utilisation impérative du dataloader canonique**, seul capable de sortir le modèle d'un effondrement sur les classes minoritaires (faisant passer la Macro Top-1 de ~5 % à plus de 20 %).
2. **Un horizon d'entraînement étendu à 15–20 époques**, nécessaire pour permettre à cette architecture de converger pleinement (le seuil de 10 époques s'avérant insuffisant).
3. **Un assouplissement de l'arrêt précoce**, en augmentant la patience à **4 ou 5 époques** afin de ne pas interrompre l'apprentissage lors des plateaux ou oscillations temporaires de la perte en début de cycle.
4. **La sauvegarde systématique du meilleur checkpoint** (basé sur la perte de validation minimale ou le score Macro Top-1 maximal, atteint vers les époques 15 à 17) plutôt que les poids de la dernière époque.