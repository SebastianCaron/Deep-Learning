import sys
import time
import gc
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, './src')
from loader import load_canonical_data
from models.custom import CustomArtCNN
from models.pretrained import ArtClassifier, ArtClassifierTinyBased
from trainer import train_one_epoch, evaluate

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")

    print("Loading Canonique dataset...")
    t0 = time.time()
    train_dt, val_dt, classes, _ = load_canonical_data(
        data_dir="./data/Art Movement/",
        train_ratio=0.8,
        seed=42,
        img_size=288
    )
    print(f"Dataset loaded in {time.time()-t0:.1f}s. Train: {len(train_dt)} images, Val: {len(val_dt)} images, Classes: {len(classes)}")

    train_loader = DataLoader(
        train_dt,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dt,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    models_to_test = [
        ("CustomArtCNN", lambda: CustomArtCNN(num_classes=len(classes))),
        ("ArtClassifier", lambda: ArtClassifier(nb_classes=len(classes))),
        ("ArtClassifierTinyBased", lambda: ArtClassifierTinyBased(nb_classes=len(classes))),
    ]

    results = {}

    for name, model_fn in models_to_test:
        print(f"\n=======================================================")
        print(f"Starting 1 epoch benchmark on RTX 3050 for: {name}")
        print(f"=======================================================")
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        model = model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        # 1 Train epoch
        t_train_start = time.time()
        train_loss, train_top1, train_top5 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler
        )
        torch.cuda.synchronize()
        t_train_end = time.time()
        train_dur = t_train_end - t_train_start
        train_fps = len(train_dt) / train_dur
        
        # 1 Eval epoch
        t_val_start = time.time()
        val_loss, micro_acc, macro_acc, top5_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion
        )
        torch.cuda.synchronize()
        t_val_end = time.time()
        val_dur = t_val_end - t_val_start
        val_fps = len(val_dt) / val_dur
        
        total_dur = train_dur + val_dur
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        res = {
            "gpu": gpu_name,
            "train_duration_sec": round(train_dur, 2),
            "train_duration_min": round(train_dur / 60, 2),
            "train_fps": round(train_fps, 2),
            "val_duration_sec": round(val_dur, 2),
            "val_fps": round(val_fps, 2),
            "total_epoch_sec": round(total_dur, 2),
            "total_epoch_min": round(total_dur / 60, 2),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "train_loss": round(train_loss, 4),
            "train_top1": round(train_top1, 2),
            "val_loss": round(val_loss, 4),
            "val_micro_top1": round(micro_acc, 2),
            "val_macro_top1": round(macro_acc, 2),
            "val_top5": round(top5_acc, 2)
        }
        results[name] = res
        print(f"\n--> {name} :")
        print(f"    Train: {train_dur:.1f}s ({train_dur/60:.2f} min, {train_fps:.1f} img/s)")
        print(f"    Val  : {val_dur:.1f}s ({val_fps:.1f} img/s)")
        print(f"    Total: {total_dur:.1f}s ({total_dur/60:.2f} min)")
        print(f"    VRAM : {peak_vram_mb:.0f} MB")
        
        del model, optimizer, criterion, scaler
        torch.cuda.empty_cache()
        gc.collect()

    print("\n\n================ FINAL RTX 3050 RESULTS ================")
    print(json.dumps(results, indent=2))
    out_file = Path("rapports/rtx3050_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")

if __name__ == '__main__':
    main()
