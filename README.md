# Deep Learning — Classification & Exploration d'oeuvres d'Art

![La Nuit étoilée, Vincent van Gogh, 1888](https://upload.wikimedia.org/wikipedia/commons/0/01/Vincent_van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg?utm_source=fr.wikipedia.org&utm_campaign=index&utm_content=original)


Projet de Deep Learning (M2) dédié à la classification de styles artistiques et à l'exploration sémantique d'oeuvres d'art à travers leurs représentations latentes, intégrant un serveur **MCP (Model Context Protocol)**.

Le pipeline s'appuie sur une taxonomie regroupant les styles en **21 mouvements artistiques** majeurs (Renaissance, Baroque, Impressionnisme, Cubisme, etc.) et extrait des embeddings 512D pour la recherche d'analogies visuelles.

---

## 1. Installation de l'environnement

### Prérequis
- Python 3.10+ (testé avec Python 3.12 / 3.14)
- Carte graphique NVIDIA recommandée avec pilotes CUDA à jour

### Création du venv
```bash
# Cloner le dépôt et se placer dans le projet
cd project

# Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate
```

### Installation des dépendances
```bash
# Dépendances de base (PyTorch, Torchvision, Pillow)
pip install -r requirements.txt

# Si vous disposez d'un GPU NVIDIA (CUDA 12.1+), installez la version GPU de PyTorch :
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dépendances pour le serveur MCP et les outils d'indexation
pip install "mcp>=2.3.0" tqdm
```

---

## 2. Téléchargement & Préparation du dataset (ART500K)

Le projet utilise le dataset **[ART500K](https://deepart.hkust.edu.hk/ART500K/art500k.html)** (~500 000 images d'oeuvres d'art et leurs métadonnées).

> **Note :** Choisissez "Raw images of visual arts with general label list: Data&Labels(56.6GB)/Labels".

### Structure attendue dans `data/`
Après extraction, le dossier `data/` doit avoir l'arborescence suivante :

```text
data/
├── Art Movement/          # Dossiers par mouvements (utilisé pour l'entraînement canonique)
├── Artists1/              # oeuvres classées par artistes (partie 1)
├── Artists2/              # oeuvres classées par artistes (partie 2)
├── Medium/                # oeuvres classées par médium/support
├── label_list.csv         # Table de métadonnées (titres, artistes, styles, chemins)
├── head_info.csv          # En-têtes des colonnes de label_list.csv
└── Readme
```

### Extraction de l'archive
Si vous disposez de l'archive `art500k.tar.gz` (téléchargement ou support externe) :
```bash
mkdir -p data
tar -xzf art500k.tar.gz -C data/
```

> **Note :** Le chargeur de données (`src/loader.py`) filtre et associe automatiquement les sous-dossiers de `data/Art Movement/` aux 21 classes canoniques sans conflit d'étiquetage.

> **Note encore :** Nous n'utilisons pas le dossier Artists1 car il n'y a quasiment pas d'informations sur le genre. Nous aurions pu cependant intégrer Artists2 mais cela ajoute des problèmatiques d'équilibrage et représentation des classes, en plus d'allonger le temps d'entrainement. Nous nous reservons le droit d'intégrer les données, de poursuivre l'entrainement et d'améliorer les models pour la V2.

---

## 3. Entraînement & Évaluation

### Modèles disponibles
- **`ArtClassifierTinyBased`** : Backbone ConvNeXt-Tiny pré-entraîné + espace latent 512D + classifieur linéaire (modèle par défaut, excellent compromis vitesse/précision).
- **`ArtClassifier`** : Backbone ResNet-50 pré-entraîné + espace latent 512D.
- **`CustomArtCNN`** : CNN personnalisé à 4 blocs convolutifs entraîné *from scratch*.

### Lancer un entraînement simple
```bash
python3 src/trainer.py
```
Les checkpoints et l'historique sont enregistrés dans le dossier `checkpoints/` :
- `best_model.pth` : meilleur modèle selon la micro-précision Top-1.
- `best_loss_model.pth` : meilleur modèle selon la perte de validation.
- `latest_model.pth` : dernier état sauvegardé.
- `history.json` : métriques d'entraînement et de validation par époque.

### Lancer la campagne de mesures & benchmarks
Pour évaluer différentes combinaisons (architectures, stratégies d'early stopping, nombre d'époques) :
```bash
python3 src/measures.py
```

---

## 4. Serveur MCP (Model Context Protocol)

Le serveur MCP (`src/serv_mcp/`) permet à un assistant IA (Claude Desktop, Cursor, Antigravity, etc.) d'interagir directement avec le modèle de vision.

### Étape 1 : Construire l'index de la galerie d'oeuvres
Avant de démarrer le serveur, extrayez les représentations latentes des oeuvres du jeu de validation :
```bash
python3 -m src.serv_mcp.build_index
```
Cette commande génère le fichier `checkpoints/gallery_index.pt`.

### Étape 2 : Lancer le serveur MCP
```bash
python3 -m src.serv_mcp.mcp_server
```

### Outils disponibles via MCP
| Outil | Description |
|---|---|
| `predict_style` | Prédit le style artistique d'une image locale ou d'un identifiant (top 5 des styles, score de confiance, métadonnées). |
| `find_similar_artworks` | Trouve les $k$ oeuvres les plus proches visuellement par similarité cosinus dans l'espace latent 512D. |
| `find_cross_style_neighbors` | Trouve des oeuvres visuellement proches mais issues d'un **autre style** (idéal pour étudier les influences croisées). |
| `get_model_info` | Fournit les détails du modèle en mémoire, la liste des classes et le volume de la galerie. |

### Configuration client (ex. Claude Desktop / Cursor / Antigravity)
Ajoutez la configuration suivante dans votre fichier de configuration MCP (`claude_desktop_config.json` ou équivalent) :

```json
{
  "mcpServers": {
    "art-explorer": {
      "command": "/chemin/absolu/vers/project/.venv/bin/python3",
      "args": ["-m", "src.serv_mcp.mcp_server"],
      "cwd": "/chemin/absolu/vers/project"
    }
  }
}
```
