import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

try:
    from src.early_stopping import EarlyStopper
    from src.loader import get_canonical_samples, get_training_transform, get_eval_transfomrm, ArtDataset
    from src.models.pretrained import ArtClassifierTinyBased
    from src.trainer import train, DEVICE
except ModuleNotFoundError:
    from early_stopping import EarlyStopper
    from loader import get_canonical_samples, get_training_transform, get_eval_transfomrm, ArtDataset
    from models.pretrained import ArtClassifierTinyBased
    from trainer import train, DEVICE


def is_run_completed(run_dir: Path) -> bool:
    final_model_path = run_dir / "final_model.pth"
    history_path = run_dir / "history.json"
    if not (final_model_path.exists() and history_path.exists()):
        return False
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return isinstance(data, list) and len(data) > 0
    except Exception:
        return False


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    n = len(values)
    mean_val = sum(values) / n
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = variance ** 0.5
    else:
        std_val = 0.0
    return {"mean": round(mean_val, 4), "std": round(std_val, 4)}


def run_cross_validation(
    n_splits: int = 3,
    data_path: str = "./data/Art Movement/",
    output_dir: str | Path = "./checkpoints/cross_val",
    nb_epochs: int = 10,
    batch_size: int = 16,
    patience: int = 2,
    lr: float = 1e-4,
    img_size: int = 288,
    num_workers: int = 4,
    seed: int = 42,
    device: str = DEVICE,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"VALIDATION CROISÉE ({n_splits}-FOLD STRATIFIÉE) - ArtClassifierTinyBased")
    print(f"Époques max : {nb_epochs} | Batch size : {batch_size} | Patience : {patience}")
    print(f"Périphérique : {device} | Checkpoints : {output_dir.resolve()}")
    print("=" * 80)

    print(f"\n[1/2] Chargement des données canoniques depuis {data_path}...")
    samples, classes, _ = get_canonical_samples(data_path)
    print(f"  Nombre total d'échantillons : {len(samples)} réparties sur {len(classes)} classes")

    y = np.array([target for _, target in samples])
    indices = np.arange(len(samples))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    summary_file = output_dir / "cv_summary.json"
    history_file = output_dir / "cv_history.json"

    fold_summaries = {}
    fold_histories = {}

    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if "folds" in saved:
                    fold_summaries = saved["folds"]
        except Exception:
            pass

    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                fold_histories = json.load(f)
        except Exception:
            pass

    train_tf = get_training_transform(img_size=img_size)
    val_tf = get_eval_transfomrm(img_size=img_size)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, y), 1):
        fold_name = f"fold_{fold_idx}"
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"PLI [{fold_idx}/{n_splits}] ({fold_name})")
        print(f"  Échantillons train : {len(train_idx)} | Val : {len(val_idx)}")
        print(f"  Dossier : {fold_dir.resolve()}")
        print("=" * 80)

        if is_run_completed(fold_dir):
            print(f"[Reprise] {fold_name} déjà complété avec succès. Chargement de l'historique...")
            with open(fold_dir / "history.json", "r", encoding="utf-8") as f:
                hist = json.load(f)
            fold_histories[fold_name] = hist
            continue

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_dt = ArtDataset(train_samples, transform=train_tf, classes=classes)
        val_dt = ArtDataset(val_samples, transform=val_tf, classes=classes)

        train_loader = DataLoader(
            train_dt,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_dt,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )

        model = ArtClassifierTinyBased(nb_classes=len(classes)).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = torch.GradScaler()
        early_stopper = EarlyStopper(patience=patience, min_delta=1e-4, mode="min")

        start_time = time.time()
        history = train(
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stopper=early_stopper,
            nb_epoch=nb_epochs,
            checkpoint_dir=fold_dir,
            restore_best_weights=True,
        )
        run_dur = time.time() - start_time

        final_payload = {
            "fold": fold_idx,
            "classes": classes,
            "run_duration_sec": run_dur,
            "final_epoch": history[-1]["epoch"] if history else nb_epochs,
            "history": history,
            "model_state_dict": model.state_dict(),
        }
        torch.save(final_payload, fold_dir / "final_model.pth")

        best_micro = max(e.get("val_micro_top1", 0.0) for e in history)
        best_macro = max(e.get("val_macro_top1", 0.0) for e in history)
        best_loss = min(e.get("val_loss", float("inf")) for e in history)
        best_top5 = max(e.get("val_top5", 0.0) for e in history)
        final_epoch_info = history[-1] if history else {}

        summary_fold = {
            "fold": fold_idx,
            "best_val_micro_top1": round(best_micro, 4),
            "best_val_macro_top1": round(best_macro, 4),
            "best_val_top5": round(best_top5, 4),
            "best_val_loss": round(best_loss, 4),
            "final_val_micro_top1": round(final_epoch_info.get("val_micro_top1", 0.0), 4),
            "final_val_loss": round(final_epoch_info.get("val_loss", 0.0), 4),
            "stopped_epoch": final_epoch_info.get("epoch", len(history)),
            "run_duration_sec": round(run_dur, 2),
        }

        fold_summaries[fold_name] = summary_fold
        fold_histories[fold_name] = history

        # Calcul des métriques globales agrégées (Moyenne ± Écart-type)
        overall_summary = {
            "n_splits": n_splits,
            "nb_epochs_max": nb_epochs,
            "batch_size": batch_size,
            "early_stopping_patience": patience,
            "best_val_micro_top1": stats([f["best_val_micro_top1"] for f in fold_summaries.values()]),
            "best_val_macro_top1": stats([f["best_val_macro_top1"] for f in fold_summaries.values()]),
            "best_val_top5": stats([f["best_val_top5"] for f in fold_summaries.values()]),
            "best_val_loss": stats([f["best_val_loss"] for f in fold_summaries.values()]),
            "final_val_micro_top1": stats([f["final_val_micro_top1"] for f in fold_summaries.values()]),
            "stopped_epoch": stats([f["stopped_epoch"] for f in fold_summaries.values()]),
            "folds": fold_summaries,
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(overall_summary, f, indent=2)

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(fold_histories, f, indent=2)

        del model, optimizer, scaler, loss_fn, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL DE LA VALIDATION CROISÉE :")
    print("-" * 80)
    for fname, fstats in fold_summaries.items():
        print(f"  {fname} : Micro Top-1 = {fstats['best_val_micro_top1']:.2f}% | Macro Top-1 = {fstats['best_val_macro_top1']:.2f}% | Top-5 = {fstats['best_val_top5']:.2f}% | Loss = {fstats['best_val_loss']:.4f} (Arrêt époque {fstats['stopped_epoch']})")
    print("-" * 80)
    print(f"  Score Micro Top-1 Global : {overall_summary['best_val_micro_top1']['mean']:.2f}% ± {overall_summary['best_val_micro_top1']['std']:.2f}%")
    print(f"  Score Macro Top-1 Global : {overall_summary['best_val_macro_top1']['mean']:.2f}% ± {overall_summary['best_val_macro_top1']['std']:.2f}%")
    print(f"  Score Top-5 Global       : {overall_summary['best_val_top5']['mean']:.2f}% ± {overall_summary['best_val_top5']['std']:.2f}%")
    print("=" * 80)

    return overall_summary, fold_histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation croisée sur ArtClassifierTinyBased")
    parser.add_argument("--folds", type=int, default=3, help="Nombre de plis (splits) pour la validation croisée")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre max d'époques")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille du batch")
    parser.add_argument("--patience", type=int, default=2, help="Patience pour early stopping")
    parser.add_argument("--data-path", type=str, default="./data/Art Movement/", help="Dossier des données")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/cross_val", help="Dossier des checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")

    args = parser.parse_args()

    run_cross_validation(
        n_splits=args.folds,
        data_path=args.data_path,
        output_dir=args.output_dir,
        nb_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed,
    )
