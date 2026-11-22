import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

try:
    from src.loader import load_canonical_data
    from src.models.pretrained import ArtClassifierTinyBased
    from src.trainer import DEVICE
except ModuleNotFoundError:
    from loader import load_canonical_data
    from models.pretrained import ArtClassifierTinyBased
    from trainer import DEVICE


def evaluate_test_set(
    data_dir: str = "./data/Art Movement/",
    checkpoint_path: str = "./checkpoints/best_model.pth",
    output_path: str = "./rapports/test_evaluation_results.json",
    batch_size: int = 32,
    device: str = DEVICE,
):
    print("=" * 80)
    print("ÉVALUATION COMPLÈTE DU MODÈLE SPÉCIALISTE SUR LE JEU DE TEST (INÉDIT — 10%)")
    print(f"Modèle : {checkpoint_path} | Périphérique : {device}")
    print("=" * 80)

    train_dt, val_dt, test_dt, classes, _ = load_canonical_data(data_dir=data_dir)
    print(f"Données : {len(train_dt)} train, {len(val_dt)} val, {len(test_dt)} test ({len(classes)} classes)")

    model = ArtClassifierTinyBased(nb_classes=len(classes)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_dt, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top5 = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            correct_top1 += (preds == targets).sum().item()
            _, top5_idx = torch.topk(outputs, 5, dim=1)
            correct_top5 += top5_idx.eq(targets.unsqueeze(1)).sum().item()
            total += targets.size(0)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_loss = running_loss / total
    micro_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    # Rappel moyen par classe (Macro Top-1)
    class_recalls = []
    for c in range(len(classes)):
        mask = all_targets == c
        if np.sum(mask) > 0:
            class_recalls.append(np.sum((all_preds == c) & mask) / np.sum(mask))
    macro_acc = 100.0 * np.mean(class_recalls)

    # One-hot targets pour AUC / mAP
    targets_onehot = np.zeros_like(all_probs)
    for idx, t in enumerate(all_targets):
        targets_onehot[idx, t] = 1.0

    classes_in_test = np.unique(all_targets)
    roc_auc = roc_auc_score(
        targets_onehot[:, classes_in_test],
        all_probs[:, classes_in_test],
        multi_class="ovr",
        average="macro",
    )
    mAP = average_precision_score(
        targets_onehot[:, classes_in_test],
        all_probs[:, classes_in_test],
        average="macro",
    )

    clf_report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, digits=4)

    results = {
        "checkpoint": str(checkpoint_path),
        "nb_test_samples": total,
        "nb_classes": len(classes),
        "test_loss": round(test_loss, 4),
        "micro_top1_acc": round(micro_acc, 2),
        "macro_top1_recall": round(macro_acc, 2),
        "top5_acc": round(top5_acc, 2),
        "macro_roc_auc_ovr": round(float(roc_auc), 4),
        "macro_mAP": round(float(mAP), 4),
        "classification_report": clf_report,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRésultats synthétiques :")
    print(f"  Test Loss                 : {test_loss:.4f}")
    print(f"  Micro Top-1               : {micro_acc:.2f}%")
    print(f"  Macro Top-1 (Macro-Rappel): {macro_acc:.2f}%")
    print(f"  Top-5                     : {top5_acc:.2f}%")
    print(f"  Macro ROC-AUC (OvR)       : {roc_auc:.4f}")
    print(f"  mAP (Macro Avg Precision) : {mAP:.4f}")
    print(f"\nRapport détaillé sauvegardé dans : {out_file.resolve()}\n")

    return results


if __name__ == "__main__":
    evaluate_test_set()
