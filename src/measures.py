import gc
import json
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from early_stopping import EarlyStopper
from loader import get_datasets, load_canonical_data
from models.custom import CustomArtCNN
from models.pretrained import ArtClassifier, ArtClassifierTinyBased
from trainer import train

# CONFIGS EDIT HERE
EARLY_STOP_VALUES = [None, 0, 2, 5]
EPOCHS = [10, 20, 30]
AVERAGE_ON = 5
DATA_PATH = "./data/Art Movement/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
BASE_CHECKPOINT_DIR = "./checkpoints"

DATA_LOADERS = {
    "simple": get_datasets,
    "canonique": load_canonical_data,
}

MODELS = {
    "CustomArtCNN": CustomArtCNN,
    "ArtClassifier": ArtClassifier,
    "ArtClassifierTinyBased": ArtClassifierTinyBased,
}


def format_early_stop(val: int | None) -> str:
    return str(val) if val is not None else "none"


def get_config_key(
    model_name: str,
    dataloader_name: str,
    nb_epochs: int,
    early_stop_val: int | None,
) -> str:
    return f"{model_name}/{dataloader_name}/{nb_epochs}/{format_early_stop(early_stop_val)}"


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


def compute_validation_summary(histories: list[list[dict]]) -> dict:
    if not histories:
        return {}

    best_micro_top1_list = []
    best_macro_top1_list = []
    best_val_loss_list = []
    best_val_top5_list = []

    final_micro_top1_list = []
    final_macro_top1_list = []
    final_val_loss_list = []
    final_val_top5_list = []
    stopped_epochs_list = []

    for run_hist in histories:
        if not run_hist:
            continue
        best_micro = max(e.get("val_micro_top1", 0.0) for e in run_hist)
        best_macro = max(e.get("val_macro_top1", 0.0) for e in run_hist)
        best_loss = min(e.get("val_loss", float("inf")) for e in run_hist)
        best_top5 = max(e.get("val_top5", 0.0) for e in run_hist)

        best_micro_top1_list.append(best_micro)
        best_macro_top1_list.append(best_macro)
        best_val_loss_list.append(best_loss)
        best_val_top5_list.append(best_top5)

        final_epoch = run_hist[-1]
        final_micro_top1_list.append(final_epoch.get("val_micro_top1", 0.0))
        final_macro_top1_list.append(final_epoch.get("val_macro_top1", 0.0))
        final_val_loss_list.append(final_epoch.get("val_loss", 0.0))
        final_val_top5_list.append(final_epoch.get("val_top5", 0.0))
        stopped_epochs_list.append(final_epoch.get("epoch", len(run_hist)))

    n = len(histories)

    return {
        "num_runs": n,
        "best_val_micro_top1": stats(best_micro_top1_list),
        "best_val_macro_top1": stats(best_macro_top1_list),
        "best_val_loss": stats(best_val_loss_list),
        "best_val_top5": stats(best_val_top5_list),
        "final_val_micro_top1": stats(final_micro_top1_list),
        "final_val_macro_top1": stats(final_macro_top1_list),
        "final_val_loss": stats(final_val_loss_list),
        "final_val_top5": stats(final_val_top5_list),
        "stopped_epoch": stats(stopped_epochs_list),
    }


def run_measures(
    epochs: list[int] = EPOCHS,
    models: dict[str, type] | None = None,
    dataloaders: dict[str, callable] | None = None,
    early_stop_values: list[int | None] = EARLY_STOP_VALUES,
    average_on: int = AVERAGE_ON,
    data_path: str = DATA_PATH,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    base_checkpoint_dir: str | Path = BASE_CHECKPOINT_DIR,
    output_history_file: str | Path | None = None,
    output_summary_file: str | Path | None = None,
    num_workers: int = 4,
) -> tuple[dict, dict]:
    if models is None:
        models = MODELS
    if dataloaders is None:
        dataloaders = DATA_LOADERS

    base_checkpoint_path = Path(base_checkpoint_dir)
    base_checkpoint_path.mkdir(parents=True, exist_ok=True)

    if output_history_file is None:
        output_history_file = base_checkpoint_path / "measures_history.json"
    else:
        output_history_file = Path(output_history_file)

    if output_summary_file is None:
        output_summary_file = base_checkpoint_path / "measures_summary.json"
    else:
        output_summary_file = Path(output_summary_file)

    dataset_cache: dict[str, tuple] = {}
    print("=" * 80)
    print("DÉMARRAGE DE LA CAMPAGNE DE MESURES")
    print(f"Périphérique : {device}")
    print(f"Modèles : {list(models.keys())}")
    print(f"Dataloaders : {list(dataloaders.keys())}")
    print(f"Époques : {epochs}")
    print(f"Early stop (patience) : {early_stop_values}")
    print(f"Répétitions par configuration : {average_on}")
    print(f"Dossier de checkpoints racine : {base_checkpoint_path.resolve()}")
    print("=" * 80)

    all_histories: dict[str, list] = {}
    if output_history_file.exists():
        try:
            with open(output_history_file, "r", encoding="utf-8") as f:
                all_histories = json.load(f)
                print(f"[Info] {len(all_histories)} configuration(s) existante(s) chargée(s) depuis {output_history_file}")
        except Exception as e:
            print(f"[Warning] Impossible de lire {output_history_file}: {e}")

    all_summaries: dict[str, dict] = {}
    if output_summary_file.exists():
        try:
            with open(output_summary_file, "r", encoding="utf-8") as f:
                all_summaries = json.load(f)
        except Exception as e:
            print(f"[Warning] Impossible de lire {output_summary_file}: {e}")

    total_configs = len(epochs) * len(models) * len(dataloaders) * len(early_stop_values)
    current_config_idx = 0

    for nb_epoch in epochs:
        for model_name, model_class in models.items():
            for dataloader_name, dataloader_fn in dataloaders.items():
                if dataloader_name not in dataset_cache:
                    print(f"\n[Chargement du dataset '{dataloader_name}' depuis {data_path}...]")
                    raw_data = dataloader_fn(data_path)
                    train_dt, val_dt, classes = raw_data[0], raw_data[1], raw_data[2]
                    dataset_cache[dataloader_name] = (train_dt, val_dt, classes)
                else:
                    train_dt, val_dt, classes = dataset_cache[dataloader_name]

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

                for early_stop in early_stop_values:
                    current_config_idx += 1
                    config_key = get_config_key(
                        model_name=model_name,
                        dataloader_name=dataloader_name,
                        nb_epochs=nb_epoch,
                        early_stop_val=early_stop,
                    )

                    config_dir = (
                        base_checkpoint_path
                        / model_name
                        / dataloader_name
                        / str(nb_epoch)
                        / format_early_stop(early_stop)
                    )
                    config_dir.mkdir(parents=True, exist_ok=True)

                    print("\n" + "=" * 80)
                    print(f"CONFIGURATION [{current_config_idx}/{total_configs}] : {config_key}")
                    print(f"Dossier de sauvegarde : {config_dir.resolve()}")
                    print("=" * 80)

                    config_histories = []

                    for run_idx in range(average_on):
                        run_label = f"run_{run_idx + 1}"
                        run_dir = config_dir / run_label
                        run_dir.mkdir(parents=True, exist_ok=True)

                        print(f"\n>>> Début {run_label}/{average_on} pour {config_key} <<<")

                        early_stopper = None
                        if early_stop is not None:
                            early_stopper = EarlyStopper(
                                patience=early_stop,
                                min_delta=1e-4,
                                mode="min",
                            )

                        model = model_class(len(classes))
                        model = model.to(device)

                        loss_fn = nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                        scaler = torch.GradScaler()

                        run_start_time = time.time()
                        history = train(
                            model=model,
                            loss=loss_fn,
                            optimizer=optimizer,
                            scaler=scaler,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            early_stopper=early_stopper,
                            nb_epoch=nb_epoch,
                            checkpoint_dir=run_dir,
                            restore_best_weights=True,
                        )
                        run_duration = time.time() - run_start_time

                        final_model_path = run_dir / "final_model.pth"
                        final_payload = {
                            "config": {
                                "model": model_name,
                                "dataloader": dataloader_name,
                                "nb_epochs": nb_epoch,
                                "early_stop": early_stop,
                                "run_idx": run_idx + 1,
                            },
                            "classes": classes,
                            "model_state_dict": model.state_dict(),
                            "final_epoch": history[-1]["epoch"] if history else nb_epoch,
                            "run_duration_sec": run_duration,
                            "history": history,
                        }
                        torch.save(final_payload, final_model_path)
                        print(f"  -> Modèle final de {run_label} sauvegardé : {final_model_path}")

                        config_histories.append(history)
                        all_histories[config_key] = config_histories

                        # Save config history and update global history immediately after each run
                        with open(config_dir / "config_history.json", "w", encoding="utf-8") as f:
                            json.dump(config_histories, f, indent=2)

                        with open(output_history_file, "w", encoding="utf-8") as f:
                            json.dump(all_histories, f, indent=2)

                        del model, optimizer, scaler, loss_fn
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                    summary = compute_validation_summary(config_histories)
                    summary["config"] = {
                        "model": model_name,
                        "dataloader": dataloader_name,
                        "nb_epochs": nb_epoch,
                        "early_stop": early_stop,
                    }
                    all_summaries[config_key] = summary

                    with open(config_dir / "config_summary.json", "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)

                    with open(output_summary_file, "w", encoding="utf-8") as f:
                        json.dump(all_summaries, f, indent=2)

                    best_micro = summary["best_val_micro_top1"]
                    final_micro = summary["final_val_micro_top1"]
                    val_loss = summary["best_val_loss"]
                    print("\n" + "-" * 60)
                    print(f"RÉSUMÉ POUR LA CONFIGURATION : {config_key}")
                    print(f"  Micro Top-1 (Meilleur) : {best_micro['mean']:.2f}% (± {best_micro['std']:.2f}%)")
                    print(f"  Micro Top-1 (Final)    : {final_micro['mean']:.2f}% (± {final_micro['std']:.2f}%)")
                    print(f"  Val Loss (Meilleure)   : {val_loss['mean']:.4f} (± {val_loss['std']:.4f})")
                    print("-" * 60)

    print("\n" + "=" * 80)
    print("CAMPAGNE DE MESURES TERMINÉE AVEC SUCCÈS")
    print(f"Historiques complets sauvegardés dans : {output_history_file.resolve()}")
    print(f"Résumés de performance sauvegardés dans : {output_summary_file.resolve()}")
    print("=" * 80)

    return all_histories, all_summaries


if __name__ == "__main__":
    run_measures()