import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader

try:
    from src.loader import load_canonical_data, get_datasets
    from src.models.pretrained import ArtClassifier, ArtClassifierTinyBased
    from src.trainer import DEVICE
except ModuleNotFoundError:
    from loader import load_canonical_data, get_datasets
    from models.pretrained import ArtClassifier, ArtClassifierTinyBased
    from trainer import DEVICE

# Palette pastel bleu cohérente pour tout le projet
PASTEL_BLUES = [
    "#D0E1FD",  # Bleu pastel très clair
    "#A3C7F7",  # Bleu pastel clair
    "#72A7EE",  # Bleu pastel moyen
    "#4285F4",  # Bleu pastel soutenu
    "#2A67D5",  # Bleu pastel foncé
    "#184CB0",  # Bleu nuit doux
]

# Colormap continue dégradé bleu pastel
PASTEL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "pastel_blue_cmap",
    ["#F8FAFC", "#E2ECF9", "#BBD6F6", "#8EBDF1", "#5C9EEA", "#357FD9", "#1E5EBA", "#123D82"],
)

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "grid.color": "#EAEAEA",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def plot_confusion_matrices(output_dir: Path):
    print("\n[1/5] Génération des matrices de confusion (Canonique & Non Canonique)...")

    # 1. Matrice Canonique (21 classes)
    _, _, test_dt, classes_canon, _ = load_canonical_data("./data/Art Movement/")
    test_loader = DataLoader(test_dt, batch_size=32, shuffle=False, num_workers=4)

    model_canon = ArtClassifierTinyBased(len(classes_canon)).to(DEVICE)
    ckpt = torch.load("checkpoints/best_model.pth", map_location=DEVICE, weights_only=False)
    model_canon.load_state_dict(ckpt["model_state_dict"])
    model_canon.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targs in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model_canon(imgs).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(targs.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes_canon)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0
    micro_acc = 100.0 * np.sum(y_true == y_pred) / len(y_true)

    fig, ax = plt.subplots(figsize=(16, 14), facecolor="white")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".0f",
        cmap=PASTEL_CMAP,
        xticklabels=classes_canon,
        yticklabels=classes_canon,
        cbar_kws={"label": "Taux de rappel (% par classe réelle)"},
        ax=ax,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="#E2E8F0",
    )
    ax.set_title(
        f"Matrice de Confusion Normalisée — Modèle Canonique (21 classes)\n"
        f"Micro Top-1 : {micro_acc:.2f}% sur l'ensemble de test ({len(y_true):,} images)",
        fontsize=14,
        fontweight="bold",
        pad=16,
        color="#1E293B",
    )
    ax.set_xlabel("Classe Prédite", fontsize=12, fontweight="bold", labelpad=10, color="#1E293B")
    ax.set_ylabel("Classe Réelle", fontsize=12, fontweight="bold", labelpad=10, color="#1E293B")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out_canon = output_dir / "confusion_matrix_canonique.png"
    plt.savefig(out_canon, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Matrice canonique sauvegardée : {out_canon}")

    # 2. Matrice Non Canonique (117 classes)
    simple_ckpt_path = Path("checkpoints/ArtClassifier/simple/10/2/run_1/best_model.pth")
    if simple_ckpt_path.exists():
        raw_simple = get_datasets("./data/Art Movement/")
        val_dt_simple = raw_simple[1]
        classes_simple = raw_simple[-1]
        val_loader_simple = DataLoader(val_dt_simple, batch_size=32, shuffle=False, num_workers=4)

        model_simple = ArtClassifier(nb_classes=len(classes_simple)).to(DEVICE)
        s_ckpt = torch.load(simple_ckpt_path, map_location=DEVICE, weights_only=False)
        model_simple.load_state_dict(s_ckpt["model_state_dict"])
        model_simple.eval()

        s_preds, s_targets = [], []
        with torch.no_grad():
            for imgs, targs in val_loader_simple:
                imgs = imgs.to(DEVICE)
                preds = model_simple(imgs).argmax(dim=1).cpu()
                s_preds.extend(preds.numpy())
                s_targets.extend(targs.numpy())

        sy_true = np.array(s_targets)
        sy_pred = np.array(s_preds)
        scm = confusion_matrix(sy_true, sy_pred, labels=range(len(classes_simple)))
        scm_norm = scm.astype("float") / (scm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0
        smicro = 100.0 * np.sum(sy_true == sy_pred) / len(sy_true)

        fig, ax = plt.subplots(figsize=(22, 18), facecolor="white")
        sns.heatmap(
            scm_norm,
            annot=False,
            cmap=PASTEL_CMAP,
            xticklabels=classes_simple,
            yticklabels=classes_simple,
            cbar_kws={"label": "Taux de rappel (% par classe réelle)"},
            ax=ax,
            vmin=0,
            vmax=100,
        )
        ax.set_title(
            f"Matrice de Confusion Normalisée — Données Brutes (117 classes)\n"
            f"Micro Top-1 : {smicro:.2f}% | Forte dispersion hors diagonale illustrant la fragmentation taxonomique",
            fontsize=15,
            fontweight="bold",
            pad=16,
            color="#1E293B",
        )
        ax.set_xlabel("Classes Prédites (117 classes)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Classes Réelles (117 classes)", fontsize=11, fontweight="bold", labelpad=8)
        plt.xticks(rotation=90, fontsize=4.5)
        plt.yticks(rotation=0, fontsize=4.5)
        plt.tight_layout()

        out_simple = output_dir / "confusion_matrix_non_canonique.png"
        plt.savefig(out_simple, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Matrice non canonique sauvegardée : {out_simple}")


def plot_models_comparison(output_dir: Path):
    print("\n[2/5] Génération du comparatif des modèles (Canonique vs Simple)...")
    models = ["CustomArtCNN\n(Scratch)", "ArtClassifier\n(ResNet-50)", "TinyBased\n(ConvNeXt-Tiny)"]
    x = np.arange(len(models))
    width = 0.22

    # Scores Canonique
    micro_canon = [67.46, 82.03, 83.26]
    macro_canon = [20.78, 55.54, 58.04]
    top5_canon = [87.99, 96.06, 96.17]

    # Scores Simple (117 classes)
    micro_simple = [58.58, 66.00, 67.54]
    macro_simple = [5.06, 11.98, 11.82]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

    # Panel 1: Canonique (21 classes)
    rects1 = ax1.bar(x - width, micro_canon, width, label="Micro Top-1", color=PASTEL_BLUES[1], edgecolor="#2B5B84", linewidth=1.2)
    rects2 = ax1.bar(x, macro_canon, width, label="Macro Top-1 (Rappel)", color=PASTEL_BLUES[3], edgecolor="#1D3E5B", linewidth=1.2)
    rects3 = ax1.bar(x + width, top5_canon, width, label="Top-5", color=PASTEL_BLUES[0], edgecolor="#4A7599", linewidth=1.2)

    ax1.set_title("Taxonomie Canonique (21 macro-classes)", fontsize=13, fontweight="bold", color="#1E293B", pad=12)
    ax1.set_ylabel("Précision (%)", fontsize=11, fontweight="bold", color="#1E293B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#1E293B",
            )

    # Panel 2: Simple (117 classes brutes)
    s_rects1 = ax2.bar(x - width / 2, micro_simple, width, label="Micro Top-1", color=PASTEL_BLUES[2], edgecolor="#2B5B84", linewidth=1.2)
    s_rects2 = ax2.bar(x + width / 2, macro_simple, width, label="Macro Top-1 (Rappel)", color=PASTEL_BLUES[4], edgecolor="#12273B", linewidth=1.2)

    ax2.set_title("Taxonomie Brute (117 classes)", fontsize=13, fontweight="bold", color="#1E293B", pad=12)
    ax2.set_ylabel("Précision (%)", fontsize=11, fontweight="bold", color="#1E293B")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    for rects in [s_rects1, s_rects2]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#1E293B",
            )

    plt.tight_layout()
    out_path = output_dir / "models_comparison_metrics.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Comparatif modèles sauvegardé : {out_path}")


def plot_convergence_curves(output_dir: Path):
    print("\n[3/5] Génération des courbes de convergence (20 époques)...")
    history_file = Path("checkpoints/measures_history.json")
    if not history_file.exists():
        print("  [Skip] measures_history.json introuvable.")
        return

    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = np.arange(1, 21)

    # Récupérer run_1 pour canonique/20/none
    convnext_hist = data.get("ArtClassifierTinyBased/canonique/20/none", [[]])[0]
    resnet_hist = data.get("ArtClassifier/canonique/20/none", [[]])[0]

    c_loss = [h["val_loss"] for h in convnext_hist] if convnext_hist else []
    c_micro = [h["val_micro_top1"] for h in convnext_hist] if convnext_hist else []
    c_macro = [h["val_macro_top1"] for h in convnext_hist] if convnext_hist else []

    r_loss = [h["val_loss"] for h in resnet_hist] if resnet_hist else []
    r_micro = [h["val_micro_top1"] for h in resnet_hist] if resnet_hist else []
    r_macro = [h["val_macro_top1"] for h in resnet_hist] if resnet_hist else []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="white")

    # 1. Perte de validation
    if c_loss and r_loss:
        ax1.plot(epochs[:len(c_loss)], c_loss, label="ConvNeXt-Tiny", color=PASTEL_BLUES[4], linewidth=2.4, marker="o", markersize=4)
        ax1.plot(epochs[:len(r_loss)], r_loss, label="ResNet-50", color=PASTEL_BLUES[2], linewidth=2.0, linestyle="--", marker="s", markersize=4)
        ax1.axvline(x=6, color="#E74C3C", linestyle=":", alpha=0.8, label="Inflexion min(loss)")
        ax1.set_title("Perte de Validation (Cross-Entropy)", fontsize=11.5, fontweight="bold", color="#1E293B")
        ax1.set_xlabel("Époque", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Loss", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=9)

    # 2. Micro Top-1
    if c_micro and r_micro:
        ax2.plot(epochs[:len(c_micro)], c_micro, label="ConvNeXt-Tiny", color=PASTEL_BLUES[3], linewidth=2.4, marker="o", markersize=4)
        ax2.plot(epochs[:len(r_micro)], r_micro, label="ResNet-50", color=PASTEL_BLUES[1], linewidth=2.0, linestyle="--", marker="s", markersize=4)
        ax2.set_title("Exactitude Micro Top-1 (%)", fontsize=11.5, fontweight="bold", color="#1E293B")
        ax2.set_xlabel("Époque", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Micro Top-1 (%)", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=9)

    # 3. Macro Top-1
    if c_macro and r_macro:
        ax3.plot(epochs[:len(c_macro)], c_macro, label="ConvNeXt-Tiny", color=PASTEL_BLUES[4], linewidth=2.4, marker="o", markersize=4)
        ax3.plot(epochs[:len(r_macro)], r_macro, label="ResNet-50", color=PASTEL_BLUES[2], linewidth=2.0, linestyle="--", marker="s", markersize=4)
        ax3.set_title("Exactitude Macro Top-1 (Rappel %)", fontsize=11.5, fontweight="bold", color="#1E293B")
        ax3.set_xlabel("Époque", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Macro Top-1 (%)", fontsize=10, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "models_convergence_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Courbes de convergence sauvegardées : {out_path}")


def plot_tradeoff_pareto(output_dir: Path):
    print("\n[4/5] Génération du compromis Performance vs Vitesse (RTX 3050)...")
    data = [
        {"name": "CustomArtCNN", "fps": 101.5, "macro": 20.78, "params": 4.96, "color": PASTEL_BLUES[1]},
        {"name": "ResNet-50", "fps": 160.3, "macro": 55.54, "params": 24.57, "color": PASTEL_BLUES[2]},
        {"name": "ConvNeXt-Tiny (Retenu)", "fps": 119.3, "macro": 58.04, "params": 28.23, "color": PASTEL_BLUES[4]},
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")

    for d in data:
        size = d["params"] * 45  # proportionnel au nombre de paramètres
        ax.scatter(d["fps"], d["macro"], s=size, color=d["color"], edgecolors="#1B4F72", linewidth=1.8, alpha=0.9, zorder=3)
        offset_y = 2.5 if d["name"] != "ResNet-50" else -3.5
        ax.annotate(
            f"{d['name']}\n({d['params']:.1f} M params, {d['fps']} img/s)",
            xy=(d["fps"], d["macro"]),
            xytext=(0, offset_y * 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#1E293B",
        )

    ax.set_title("Compromis Débit d'Inférence vs Macro Top-1 (NVIDIA RTX 3050)", fontsize=13, fontweight="bold", pad=14, color="#1E293B")
    ax.set_xlabel("Débit d'Inférence en Validation (images / seconde)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Macro Top-1 (Rappel Moyen, %)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlim(80, 180)
    ax.set_ylim(10, 68)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Légende pour la taille des bulles
    p1 = ax.scatter([], [], s=5 * 45, color="#D0E1FD", edgecolors="#1B4F72", label="~5 M params")
    p2 = ax.scatter([], [], s=25 * 45, color="#72A7EE", edgecolors="#1B4F72", label="~25 M params")
    ax.legend(scatterpoints=1, frameon=True, title="Taille du modèle", loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "models_tradeoff.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Compromis vitesse/performance sauvegardé : {out_path}")


def plot_test_breakdown_and_mcp(output_dir: Path):
    print("\n[5/5] Génération des graphiques du test set et de la Precision@k MCP...")

    # A. Breakdown Test Set (F1 & Rappel par classe)
    test_json = Path("rapports/test_evaluation_results.json")
    if test_json.exists():
        with open(test_json, "r", encoding="utf-8") as f:
            t_data = json.load(f)

        report = t_data.get("classification_report", {})
        classes_data = []
        for k, v in report.items():
            if k not in ["accuracy", "macro avg", "weighted avg"] and isinstance(v, dict):
                classes_data.append({
                    "class": k,
                    "f1": v["f1-score"] * 100.0,
                    "recall": v["recall"] * 100.0,
                    "support": v["support"],
                })

        classes_data.sort(key=lambda item: item["f1"], reverse=True)
        c_names = [item["class"] for item in classes_data]
        f1_vals = [item["f1"] for item in classes_data]
        rec_vals = [item["recall"] for item in classes_data]

        y_pos = np.arange(len(c_names))
        fig, ax = plt.subplots(figsize=(12, 8.5), facecolor="white")
        h = 0.38

        b1 = ax.barh(y_pos - h / 2, f1_vals, h, label="F1-Score (%)", color=PASTEL_BLUES[3], edgecolor="#1D3E5B", linewidth=1.0)
        b2 = ax.barh(y_pos + h / 2, rec_vals, h, label="Rappel (%)", color=PASTEL_BLUES[1], edgecolor="#2B5B84", linewidth=1.0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(c_names, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel("Score (%) sur le jeu de test inédit (1 663 images)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title(
            f"Performances Détaillées par Mouvement Artistique (Jeu de Test Inédit)\n"
            f"Macro F1 : {report.get('macro avg', {}).get('f1-score', 0)*100:.2f}% | "
            f"ROC-AUC : {t_data.get('macro_roc_auc_ovr', 0):.4f} | "
            f"mAP : {t_data.get('macro_mAP', 0):.4f}",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#1E293B",
        )
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        out_test = output_dir / "test_metrics_breakdown.png"
        plt.savefig(out_test, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Breakdown test set sauvegardé : {out_test}")

    # B. Precision@k MCP
    mcp_json = Path("rapports/mcp_precision_at_k.json")
    if mcp_json.exists():
        with open(mcp_json, "r", encoding="utf-8") as f:
            mcp_data = json.load(f)

        metrics = mcp_data.get("metrics", {})
        k_labels = list(metrics.keys())
        k_vals = list(metrics.values())

        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
        bars = ax.bar(k_labels, k_vals, color=PASTEL_BLUES[:len(k_labels)], edgecolor="#1D3E5B", linewidth=1.2, width=0.5)

        ax.set_title(
            f"Évaluation de la Recherche par Similarité MCP (Precision@k)\n"
            f"Mesurée sur {mcp_data.get('nb_query_samples', 0)} requêtes du jeu de test inédit",
            fontsize=12.5,
            fontweight="bold",
            pad=14,
            color="#1E293B",
        )
        ax.set_ylabel("Pertinence du style voisin (% partageant le même mouvement)", fontsize=10.5, fontweight="bold")
        ax.set_ylim(60, 85)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color="#1E293B",
            )

        plt.tight_layout()
        out_mcp = output_dir / "mcp_precision_at_k.png"
        plt.savefig(out_mcp, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Graphique Precision@k MCP sauvegardé : {out_mcp}")


def main():
    print("=" * 80)
    print("GÉNÉRATION COMPLÈTE DES VISUALISATIONS (STYLE : DÉGRADÉ BLEU PASTEL)")
    print("=" * 80)
    output_dir = Path("./rapports")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrices(output_dir)
    plot_models_comparison(output_dir)
    plot_convergence_curves(output_dir)
    plot_tradeoff_pareto(output_dir)
    plot_test_breakdown_and_mcp(output_dir)

    print("\n" + "=" * 80)
    print("TOUTES LES FIGURES ONT ÉTÉ GÉNÉRÉES AVEC SUCCÈS DANS rapports/ !")
    print("=" * 80)


if __name__ == "__main__":
    main()
