import argparse
import gc
import json
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.early_stopping import EarlyStopper
    from src.loader import load_canonical_data
    from src.models.pretrained import ArtClassifierTinyBased
    from src.trainer import train, evaluate, DEVICE
except ModuleNotFoundError:
    from early_stopping import EarlyStopper
    from loader import load_canonical_data
    from models.pretrained import ArtClassifierTinyBased
    from trainer import train, evaluate, DEVICE

ABLATION_CONFIGS = {
    "baseline": {
        "description": "Modèle complet de référence (ConvNeXt-Tiny + Latent Space 512 + BN + GELU + Dropout 0.4)",
        "model_kwargs": {
            "use_bn": True,
            "use_latent": True,
            "dropout": 0.4,
            "activation": "gelu",
            "freeze_backbone": False,
        },
    },
    "no_batchnorm": {
        "description": "Ablation de la BatchNorm1d dans l'espace latent",
        "model_kwargs": {
            "use_bn": False,
            "use_latent": True,
            "dropout": 0.4,
            "activation": "gelu",
            "freeze_backbone": False,
        },
    },
    "no_dropout": {
        "description": "Ablation du Dropout (p=0.0) dans l'espace latent",
        "model_kwargs": {
            "use_bn": True,
            "use_latent": True,
            "dropout": 0.0,
            "activation": "gelu",
            "freeze_backbone": False,
        },
    },
    "no_latent_space": {
        "description": "Ablation du goulot d'étranglement latent (Classifieur linéaire direct 768 -> nb_classes)",
        "model_kwargs": {
            "use_bn": False,
            "use_latent": False,
            "dropout": 0.0,
            "freeze_backbone": False,
        },
    },
    "frozen_backbone": {
        "description": "Ablation du fine-tuning (Backbone ConvNeXt-Tiny gelé, entraînement de la tête uniquement)",
        "model_kwargs": {
            "use_bn": True,
            "use_latent": True,
            "dropout": 0.4,
            "activation": "gelu",
            "freeze_backbone": True,
        },
    },
    "relu_activation": {
        "description": "Remplacement de l'activation GELU par ReLU dans l'espace latent",
        "model_kwargs": {
            "use_bn": True,
            "use_latent": True,
            "dropout": 0.4,
            "activation": "relu",
            "freeze_backbone": False,
        },
    },
}


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


def run_ablation(
    configs: list[str] | None = None,
    data_path: str = "./data/Art Movement/",
    output_dir: str | Path = "./checkpoints/ablation",
    nb_epochs: int = 10,
    batch_size: int = 16,
    patience: int = 2,
    lr: float = 1e-4,
    num_workers: int = 4,
    device: str = DEVICE,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())

    print("=" * 80)
    print("CAMPAGNE D'ABLATION - ArtClassifierTinyBased")
    print(f"Configurations : {configs}")
    print(f"Époques max : {nb_epochs} | Batch size : {batch_size} | Patience : {patience}")
    print(f"Périphérique : {device} | Checkpoints : {output_dir.resolve()}")
    print("=" * 80)

    print(f"\n[1/2] Chargement des données canoniques depuis {data_path}...")
    train_dt, val_dt, classes, _ = load_canonical_data(data_path)
    print(f"  Train : {len(train_dt)} images | Val : {len(val_dt)} images | Classes : {len(classes)}")

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

    summary_file = output_dir / "ablation_summary.json"
    history_file = output_dir / "ablation_history.json"

    all_summaries = {}
    all_histories = {}

    if summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                all_summaries = json.load(f)
        except Exception:
            pass

    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                all_histories = json.load(f)
        except Exception:
            pass

    total_configs = len(configs)

    for idx, cfg_name in enumerate(configs, 1):
        if cfg_name not in ABLATION_CONFIGS:
            print(f"[Attention] Configuration '{cfg_name}' inconnue, ignorée.")
            continue

        cfg_info = ABLATION_CONFIGS[cfg_name]
        cfg_dir = output_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[{idx}/{total_configs}] CONFIGURATION D'ABLATION : {cfg_name}")
        print(f"Description : {cfg_info['description']}")
        print(f"Paramètres : {cfg_info['model_kwargs']}")
        print("=" * 80)

        if is_run_completed(cfg_dir):
            print(f"[Reprise] Configuration '{cfg_name}' déjà terminée. Chargement de l'historique...")
            with open(cfg_dir / "history.json", "r", encoding="utf-8") as f:
                history = json.load(f)
            all_histories[cfg_name] = history
            continue

        # Instanciation du modèle avec les variantes d'ablation
        model = ArtClassifierTinyBased(
            nb_classes=len(classes),
            **cfg_info["model_kwargs"],
        ).to(device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Paramètres entraînables : {trainable_params:,} / {total_params:,}")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
        )
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
            checkpoint_dir=cfg_dir,
            restore_best_weights=True,
        )
        run_dur = time.time() - start_time

        final_payload = {
            "config_name": cfg_name,
            "description": cfg_info["description"],
            "model_kwargs": cfg_info["model_kwargs"],
            "classes": classes,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "run_duration_sec": run_dur,
            "final_epoch": history[-1]["epoch"] if history else nb_epochs,
            "history": history,
            "model_state_dict": model.state_dict(),
        }
        torch.save(final_payload, cfg_dir / "final_model.pth")

        # Résumé des performances pour cette configuration
        best_micro = max(e.get("val_micro_top1", 0.0) for e in history)
        best_macro = max(e.get("val_macro_top1", 0.0) for e in history)
        best_loss = min(e.get("val_loss", float("inf")) for e in history)
        best_top5 = max(e.get("val_top5", 0.0) for e in history)
        final_epoch_info = history[-1] if history else {}

        cfg_summary = {
            "config_name": cfg_name,
            "description": cfg_info["description"],
            "model_kwargs": cfg_info["model_kwargs"],
            "trainable_params": trainable_params,
            "best_val_micro_top1": round(best_micro, 4),
            "best_val_macro_top1": round(best_macro, 4),
            "best_val_top5": round(best_top5, 4),
            "best_val_loss": round(best_loss, 4),
            "final_val_micro_top1": round(final_epoch_info.get("val_micro_top1", 0.0), 4),
            "final_val_loss": round(final_epoch_info.get("val_loss", 0.0), 4),
            "stopped_epoch": final_epoch_info.get("epoch", len(history)),
            "run_duration_sec": round(run_dur, 2),
        }

        all_summaries[cfg_name] = cfg_summary
        all_histories[cfg_name] = history

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2)

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(all_histories, f, indent=2)

        del model, optimizer, scaler, loss_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL DE LA CAMPAGNE D'ABLATION :")
    print("-" * 80)
    print(f"{'Configuration':<18} | {'Micro Top-1':<12} | {'Macro Top-1':<12} | {'Top-5':<8} | {'Best Loss':<10} | {'Époque'}")
    print("-" * 80)
    for name, s in all_summaries.items():
        print(f"{name:<18} | {s['best_val_micro_top1']:5.2f}%       | {s['best_val_macro_top1']:5.2f}%       | {s['best_val_top5']:5.2f}%  | {s['best_val_loss']:.4f}     | {s['stopped_epoch']}")
    print("=" * 80)

    return all_summaries, all_histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests par ablation sur ArtClassifierTinyBased")
    parser.add_argument("--configs", nargs="+", default=["baseline", "no_batchnorm", "no_dropout", "no_latent_space", "frozen_backbone"], help="Configurations à tester")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre max d'époques")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille du batch")
    parser.add_argument("--patience", type=int, default=2, help="Patience pour early stopping")
    parser.add_argument("--data-path", type=str, default="./data/Art Movement/", help="Dossier des données")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/ablation", help="Dossier des checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    run_ablation(
        configs=args.configs,
        data_path=args.data_path,
        output_dir=args.output_dir,
        nb_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
    )
