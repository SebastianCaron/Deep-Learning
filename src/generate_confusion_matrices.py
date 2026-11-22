from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader

from loader import load_canonical_data, get_datasets
from models.pretrained import ArtClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(model, loader, device=DEVICE):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix_canonical(
    y_true,
    y_pred,
    classes,
    output_path: Path,
    title: str = "Matrice de Confusion — Données Canoniques (21 classes)",
):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    # Normalisation par ligne (Recall pour chaque classe réelle)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0

    acc = 100.0 * np.sum(y_true == y_pred) / len(y_true)

    plt.figure(figsize=(18, 15))
    sns.set_theme(style="white")

    # Formatage des annotations : pourcentage lisible
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            if val >= 1.0:
                annot_matrix[i, j] = f"{val:.0f}%"
            elif val > 0.0:
                annot_matrix[i, j] = f"{val:.1f}%"
            else:
                annot_matrix[i, j] = ""

    ax = sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Taux de rappel (% par classe réelle)"},
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="#dddddd",
        annot_kws={"size": 8, "va": "center", "ha": "center"},
        vmin=0,
        vmax=100,
    )

    ax.set_title(f"{title}\nPrécision globale (Micro Top-1) : {acc:.2f}% | Total : {len(y_true):,} images de test", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Mouvement Prédit", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("Mouvement Réel", fontsize=13, fontweight="bold", labelpad=12)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Matrice canonique sauvegardée avec succès : {output_path.resolve()}")


def plot_confusion_matrix_non_canonical(
    y_true,
    y_pred,
    classes,
    output_path: Path,
    title: str = "Matrice de Confusion — Données Non Canoniques (117 classes)",
):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    # Normalisation par ligne
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0

    acc = 100.0 * np.sum(y_true == y_pred) / len(y_true)

    # Grande figure haute résolution pour 117 classes
    plt.figure(figsize=(24, 20))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        cm_norm,
        annot=False,  # 117x117 cellules rendrait le texte illisible
        cmap="crest",
        cbar_kws={"label": "Taux de rappel (% par classe réelle)"},
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=100,
    )

    ax.set_title(
        f"{title}\nPrécision globale (Micro Top-1) : {acc:.2f}% | Total : {len(y_true):,} images de test\n(Remarque : dispersion hors-diagonale illustrant les confusions entre sous-courants redondants)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Classe Prédite (117 classes brutes)", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("Classe Réelle (117 classes brutes)", fontsize=13, fontweight="bold", labelpad=12)

    plt.xticks(rotation=90, ha="center", fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Matrice non canonique sauvegardée avec succès : {output_path.resolve()}")


def main():
    rapports_dir = Path("./rapports")
    rapports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GÉNÉRATION DES GRILLES DE CONFUSION (CANONIQUE & NON CANONIQUE)")
    print(f"Périphérique : {DEVICE}")
    print("=" * 80)

    # 1. Modèle Canonique
    print("\n[1/2] Traitement du modèle entraîné sur données canoniques (21 classes)...")
    _, val_dt_canon, classes_canon, _ = load_canonical_data("./data/Art Movement/")
    val_loader_canon = DataLoader(val_dt_canon, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    canon_ckpt_path = Path("checkpoints/ArtClassifier/canonique/10/2/run_1/best_model.pth")
    if not canon_ckpt_path.exists():
        canon_ckpt_path = Path("checkpoints/ArtClassifier/canonique/10/2/run_1/best_loss_model.pth")

    print(f"  Chargement des poids : {canon_ckpt_path}...")
    canon_ckpt = torch.load(canon_ckpt_path, map_location=DEVICE, weights_only=False)
    model_canon = ArtClassifier(nb_classes=len(classes_canon)).to(DEVICE)
    model_canon.load_state_dict(canon_ckpt["model_state_dict"])

    print("  Évaluation sur le jeu de validation canonique...")
    y_pred_canon, y_true_canon = evaluate_model(model_canon, val_loader_canon)

    out_canon = rapports_dir / "confusion_matrix_canonique.png"
    plot_confusion_matrix_canonical(y_true_canon, y_pred_canon, classes_canon, out_canon)
    del model_canon, val_loader_canon, val_dt_canon
    torch.cuda.empty_cache()

    # 2. Modèle Non Canonique
    print("\n[2/2] Traitement du modèle entraîné sur données non canoniques (117 classes)...")
    _, val_dt_simple, classes_simple = get_datasets("./data/Art Movement/")
    val_loader_simple = DataLoader(val_dt_simple, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    simple_ckpt_path = Path("checkpoints/ArtClassifier/simple/10/2/run_1/best_model.pth")
    if not simple_ckpt_path.exists():
        simple_ckpt_path = Path("checkpoints/ArtClassifier/simple/10/2/run_1/best_loss_model.pth")

    print(f"  Chargement des poids : {simple_ckpt_path}...")
    simple_ckpt = torch.load(simple_ckpt_path, map_location=DEVICE, weights_only=False)
    model_simple = ArtClassifier(nb_classes=len(classes_simple)).to(DEVICE)
    model_simple.load_state_dict(simple_ckpt["model_state_dict"])

    print("  Évaluation sur le jeu de validation non canonique...")
    y_pred_simple, y_true_simple = evaluate_model(model_simple, val_loader_simple)

    out_simple = rapports_dir / "confusion_matrix_non_canonique.png"
    plot_confusion_matrix_non_canonical(y_true_simple, y_pred_simple, classes_simple, out_simple)
    del model_simple, val_loader_simple, val_dt_simple
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("GÉNÉRATION TERMINÉE AVEC SUCCÈS !")
    print(f"  1. Image canonique     : {out_canon.resolve()}")
    print(f"  2. Image non canonique : {out_simple.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
